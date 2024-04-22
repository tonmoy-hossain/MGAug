def Mgridplot(u, Hpath, Nx=64, Ny=64, displacement=True, color='red', dpi=128, scale=1, **kwargs):
    """Given a displacement field, plot a displaced grid"""
    u = to_numpy(u)

    assert u.shape[0] == 1, "Only send one deformation at a time"
   
    # plt.figure(dpi= 128)
    plt.figure(figsize=(1,1))
    plt.xticks([])  
    plt.yticks([])  
    plt.axis('off')  
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.margins(0,0)
    

    if Nx is None:
        Nx = u.shape[2]
    if Ny is None:
        Ny = u.shape[3]
    # downsample displacements
    h = np.copy(u[0,:,::u.shape[2]//Nx, ::u.shape[3]//Ny])

    # now reset to actual Nx Ny that we achieved
    Nx = h.shape[1]
    Ny = h.shape[2]
    # adjust displacements for downsampling
    h[0,...] /= float(u.shape[2])/Nx
    h[1,...] /= float(u.shape[3])/Ny

    if displacement: # add identity
        '''
            h[0]: 
        '''
        h[0,...] += np.arange(Nx).reshape((Nx,1))  #h[0]:  (118, 109)  add element: 118*1
        h[1,...] += np.arange(Ny).reshape((1,Ny))

    # put back into original index space
    h[0,...] *= float(u.shape[2])/Nx
    h[1,...] *= float(u.shape[3])/Ny
    # create a meshgrid of locations
    for i in range(h.shape[1]):
        plt.plot( h[0,i,:], h[1,i,:], color=color, linewidth=0.2, **kwargs)
    for i in range(h.shape[2]):
        plt.plot(h[0,:,i], h[1,:,i],  color=color, linewidth=0.2, **kwargs)
    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.savefig(Hpath,dpi= dpi*scale,transparent=False)
    plt.show()
    plt.cla()
    plt.clf()
    plt.close()
    plt.close('all')





    ##PDphi_t:   1,2,64,64
    Mgridplot(PDphi_t, "", int(imagesize/2), int(imagesize/2), False,  dpi=imagesize, scale=2)