from operator import pos
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.distributions.normal import Normal
import torch.optim as optim
from layers import *
from modelio import LoadableModel, store_config_args

from gmvae_utils import *

# Inference Network
class InferenceNet(nn.Module):
  def __init__(self, x_dim, z_dim, y_dim):
    super(InferenceNet, self).__init__()
    latent_dim = 64

    # q(y|x)
    self.inference_qyx = torch.nn.ModuleList([
        nn.Linear(x_dim, latent_dim),
        nn.ReLU(),
        nn.Linear(latent_dim, latent_dim),
        nn.ReLU(),
        GumbelSoftmax(latent_dim, y_dim)
    ])

    # q(z|y,x)
    self.inference_qzyx = torch.nn.ModuleList([
        nn.Linear(x_dim + y_dim, latent_dim),
        nn.ReLU(),
        nn.Linear(latent_dim, latent_dim),
        nn.ReLU(),
        Gaussian(latent_dim, z_dim)
    ])

  # q(y|x)
  def qyx(self, x, temperature, hard):
    num_layers = len(self.inference_qyx)
    for i, layer in enumerate(self.inference_qyx):
      if i == num_layers - 1:
        #last layer is gumbel softmax
        x = layer(x, temperature, hard)
      else:
        x = layer(x)
    return x

  # q(z|x,y)
  def qzxy(self, x, y):
    concat = torch.cat((x, y), dim=1)  
    for layer in self.inference_qzyx:
      concat = layer(concat)
    return concat
  
  def forward(self, x, temperature=1.0, hard=0):
    #x = Flatten(x)

    # q(y|x)
    logits, prob, y = self.qyx(x, temperature, hard)
    
    # q(z|x,y)
    mu, var, z = self.qzxy(x, y)

    output = {'mean': mu, 'var': var, 'gaussian': z, 
              'logits': logits, 'prob_cat': prob, 'categorical': y}
    return output

# Generative Network
class GenerativeNet(nn.Module):
  def __init__(self, x_dim, z_dim, y_dim):
    super(GenerativeNet, self).__init__()

    latent_dim = 64
    # p(z|y)
    self.y_mu = nn.Linear(y_dim, z_dim)
    self.y_var = nn.Linear(y_dim, z_dim)

    # p(x|z)
    self.generative_pxz = torch.nn.ModuleList([
        nn.Linear(z_dim, latent_dim),
        nn.ReLU(),
        nn.Linear(latent_dim, latent_dim),
        nn.ReLU(),
        nn.Linear(latent_dim, x_dim),
        torch.nn.Sigmoid()
    ])

  # p(z|y)
  def pzy(self, y):
    y_mu = self.y_mu(y)
    y_var = F.softplus(self.y_var(y))
    return y_mu, y_var

  # p(x|z)
  def pxz(self, z):
    for layer in self.generative_pxz:
      z = layer(z)
    return z

  def forward(self, z, y):
    # p(z|y)
    y_mu, y_var = self.pzy(y)

    # p(x|z)
    x_rec = self.pxz(z)

    output = {'y_mean': y_mu, 'y_var': y_var, 'x_rec': x_rec}
    return output


class GMVAENet(nn.Module):
  def __init__(self, x_dim, z_dim, y_dim):
    super(GMVAENet, self).__init__()

    self.inference = InferenceNet(x_dim, z_dim, y_dim)
    self.generative = GenerativeNet(x_dim, z_dim, y_dim)

    # weight initialization
    for m in self.modules():
      if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias.data is not None:
          init.constant_(m.bias, 0) 

  def forward(self, x, temperature=1.0, hard=0):
    
    out_inf = self.inference(x, temperature, hard)
    z, y = out_inf['gaussian'], out_inf['categorical']
    out_gen = self.generative(z, y)

    # merge output
    output = out_inf
    for key, value in out_gen.items():
      output[key] = value
    return output


class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """
    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """

        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]
        
        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf)) #nf = prev_nf
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, nf, nf)) #prev_nf = nf
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

#         # configure encoder (down-sampling path)
#         prev_nf = infeats
#         encoder_nfs = [prev_nf]
#         self.encoder = nn.ModuleList()
#         for level in range(self.nb_levels - 1):
#             convs = nn.ModuleList()
#             for conv in range(nb_conv_per_level):
#                 nf = enc_nf[level * nb_conv_per_level + conv]
#                 convs.append(ConvBlock(ndims, prev_nf, nf))
#                 prev_nf = nf
#             self.encoder.append(convs)
#             encoder_nfs.append(prev_nf)

#         # configure decoder (up-sampling path)
#         encoder_nfs = np.flip(encoder_nfs)
#         self.decoder = nn.ModuleList()
#         for level in range(self.nb_levels - 1):
#             convs = nn.ModuleList()
#             for conv in range(nb_conv_per_level):
#                 nf = dec_nf[level * nb_conv_per_level + conv]
#                 convs.append(ConvBlock(ndims, prev_nf, nf))
#                 prev_nf = nf
#             self.decoder.append(convs)
#             if not half_res or level < (self.nb_levels - 2):
#                 prev_nf += encoder_nfs[level]

#         # now we take care of any remaining convolutions
#         self.remaining = nn.ModuleList()
#         for num, nf in enumerate(final_convs):
#             self.remaining.append(ConvBlock(ndims, prev_nf, nf))
#             prev_nf = nf

#         # cache final number of features
#         self.final_nf = prev_nf

        self.losses = LossFunctions()
        
        self.model = GMVAENet(32*32*32, 64, 2).to("cuda")

    def forward(self, x):
        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)
        latent = x
        
        data = x.view(x.size(0), -1)
        
        self.model.train()
        out_net = self.model(data)
    
        # obtain network variables
        z, data_recon = out_net['gaussian'], out_net['x_rec'] 
        logits, prob_cat = out_net['logits'], out_net['prob_cat']
        y_mu, y_var = out_net['y_mean'], out_net['y_var']
        mu, var = out_net['mean'], out_net['var']
        
        # reconstruction loss
        loss_rec = self.losses.reconstruction_loss(data, data_recon, 'mse')
        
        # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
        # KLD = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
        
        # gaussian loss
        loss_gauss = self.losses.gaussian_loss(z, mu, var, y_mu, y_var)

        # categorical loss
        loss_cat = -self.losses.entropy(logits, prob_cat) - np.log(0.1)

        # total loss
        loss_total = 1 * loss_gauss + 1 * loss_cat + 1 * loss_rec #1 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
        
        # data = self.reparameterize(data, data)
        x = data_recon.view(-1, 32, 32, 32)
        
        
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
                
        x = self.upsampling[level](x)
        
#         for level, convs in enumerate(self.decoder):
#             for conv in convs:
#                 x = conv(x)
#             if not self.half_res or level < (self.nb_levels - 2):

#                 x = self.upsampling[level](x)
#                 x = torch.cat([x, x_history.pop()], dim=1)
                
        
        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)
            
        return x, latent, loss_total


class DiffeoDense(LoadableModel):
    @store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False):

        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            infeats=(src_feats + trg_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                'Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers (downsize)
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = SpatialTransformer(inshape)
        
        
    def forward(self, source, target, registration=False):

        x = torch.cat([source, target], dim=1)
        
        x,latent_f, gmvae_loss = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)
        

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow
        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None
        
        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None
        
        # return non-integrated flow field if training
        if not registration:
            return (y_source, y_target, preint_flow) if self.bidir else (y_source, preint_flow)
        else:
            return y_source, pos_flow, latent_f, gmvae_loss


class ConvBlock(nn.Module):

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out
