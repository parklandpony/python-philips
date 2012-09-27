# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 14:01:08 2011

@author: Eric
"""

def read_philips_list(filename,phasefilename='',fieldmapname='',sensefilename='',coords='',readtype='new'):
    from os.path import split
    from nibabel import load
    from numpy import flipud, array, rot90, fliplr, shape, zeros, diag, eye, dot, ones, meshgrid, linspace, expand_dims
    from numpy.linalg import inv, lstsq
    from numpy.fft import ifftshift, ifftn
    from scipy.signal import medfilt    
    from scipy.ndimage.interpolation import affine_transform
    
    f=open(filename,'r',).readlines()
    scan={}
    scan['parameterfilename']=filename
    scan['datafilename']=filename[:-4]+'data'
    scan['pathname'],scan['basename']=split(filename)
    #print scan['parameterfilename']
    #print scan['datafilename']
    scan['scaninfo']={}        
    scan['datainfo']={}
    scan['datainfo']['STD']=[] #Standard data vector (image data or spectroscopy data)
    scan['datainfo']['REJ']=[] #Rejected standard data vector
    scan['datainfo']['PHX']=[] #Correction data vector for EPI/GraSE phase correction
    scan['datainfo']['FRX']=[] #Correction data vector for frequency spectrum correction
    scan['datainfo']['NOI']=[] #Preparation phase data vector for noise determination
    scan['datainfo']['NAV']=[] #Phase navigator data vector
    scan['scaninfo']['number_of_coil_channels']={}
    scan['scaninfo']['number_of_coil_channels'][0]=1 #because it doesn't always exist
    scan['scaninfo']['kx_range']=[0,0]
    scan['scaninfo']['ky_range']=[0,0]
    scan['scaninfo']['kz_range']=[0,0]
    scan['scaninfo']['kx_oversample_factor']=1
    for line in f:
        if line[0]!='#': #if the line isn't a comment
            #print line
            splitline=line.split()
            if splitline[0]=='.': #global scan data
                #print len(splitline)
                if len(splitline)==7: #if there is no start and end
                    try: #assume integers
                        scan['scaninfo'][splitline[4]]=int(splitline[6])
                    except: #if they are floats
                        scan['scaninfo'][splitline[4]]=float(splitline[6])
                elif len(splitline)==10: #for the stupid channel number
                    #print splitline
                    scan['scaninfo'][splitline[4]+'_'+splitline[5]+'_'+splitline[6]+'_'+splitline[7]][int(splitline[3])]=int(splitline[9])
                elif len(splitline)==9: #for the SENSE factor
                    scan['scaninfo'][splitline[4]+'_'+splitline[5]+'_'+splitline[6]]=float(splitline[8])
                else:
                    #print line
                    scan['scaninfo'][splitline[4]]=(int(splitline[6]),int(splitline[7]))
            else:
                #print splitline
                scan['datainfo'][splitline[0]].append([int(i) for i in splitline[1:]])
                #print line.split()
    #this is the list of parameters that you get!
    #mix   dyn   card  echo  loca  chan  extr1 extr2 ky    kz    n.a.  aver  sign  rf    grad  enc   rtop  rr    size   offset
    
    #figure out the parameters of the scan for easier reading (both data and human)
    if scan['scaninfo']['kx_range'][0]==scan['scaninfo']['kx_range'][1]: #if cpx data
        scan['scaninfo']['ifcpx']=1
        scan['scaninfo']['xsize']=scan['scaninfo']['X-resolution']
        scan['scaninfo']['ysize']=scan['scaninfo']['Y-resolution']
        scan['scaninfo']['zsize']=scan['scaninfo']['Z-resolution']
    else:
        scan['scaninfo']['ifcpx']=0
        scan['scaninfo']['xsize']=(scan['scaninfo']['kx_range'][1]+1)*2#scan['scaninfo']['kx_range'][1]-scan['scaninfo']['kx_range'][0]+1
        scan['scaninfo']['ysize']=scan['scaninfo']['ky_range'][1]*2+1#scan['scaninfo']['ky_range'][1]-scan['scaninfo']['ky_range'][0]+1
        if scan['scaninfo']['ky_range'][1]-scan['scaninfo']['ky_range'][0]+1 !=scan['scaninfo']['ysize']:
            scan['partialfourier']=1
        else:
            scan['partialfourier']=0
        scan['scaninfo']['zsize']=scan['scaninfo']['kz_range'][1]*2+1#scan['scaninfo']['kz_range'][1]-scan['scaninfo']['kz_range'][0]+1
        
    #print scan['scaninfo']['number_of_coil_channels']
    #print 1 in scan['scaninfo']['number_of_coil_channels']
    # the refscan for coil sensitivity is complicated to parse...
    if (1 in scan['scaninfo']['number_of_coil_channels']) and scan['scaninfo']['number_of_coil_channels'][0]!=scan['scaninfo']['number_of_coil_channels'][1]:
        scan['scaninfo']['ifrefscan']=1
        #ncoils=scan['scaninfo']['number_of_coil_channels'][0]+1 #special case for refscan
        #nslices=scan['scaninfo']['number_of_locations']-1
        #refscan_flag=1
    else:
        scan['scaninfo']['ifrefscan']=0
    scan['scaninfo']['ncoils']=scan['scaninfo']['number_of_coil_channels'][0] #note that these need to be edited in below in some cases...
    scan['scaninfo']['nslices']=scan['scaninfo']['number_of_locations']
        #refscan_flag=0
    #print refscan_flag
    scan['scaninfo']['ndynamics']=scan['scaninfo']['number_of_dynamic_scans']
    scan['scaninfo']['naverages']=scan['scaninfo']['number_of_signal_averages'] #averages are just averages (it looks like there are 2 b=0 images by default)
    scan['scaninfo']['nextra1']=scan['scaninfo']['number_of_extra_attribute_1_values'] #this looks like the number of diffusion values (including b=0)
    scan['scaninfo']['nextra2']=scan['scaninfo']['number_of_extra_attribute_2_values'] #this looks like the number of diffusion directions (NOT including b=0)
    
    try:
        scan['scaninfo']['sense_factor']=max(scan['scaninfo']['Y-direction_SENSE_factor'],scan['scaninfo']['X-direction_SENSE_factor'])
    except:
        scan['scaninfo']['sense_factor']=1
    
    
    
    if readtype=='large':
        scan=read_philips_list_raw(scan)
    else:
        scan=read_philips_dic_raw(scan)
        #scan=read_philips_dic_fiddle(scan)
        
        
    #and lets read any external phase correction data
    scan['datainfo']['auxiliary_phase_data']={}
    if phasefilename!='':
        print('-------------reading external phase file-------------------------')
        scan['datainfo']['auxiliary_phase_data']=read_philips_list(phasefilename)
        print('-------------completed reading external phase file---------------')
    scan['scaninfo']['auxiliary_phase_data_filename']=phasefilename
    
    if fieldmapname!='':
        print('Reading fieldmap file '+fieldmapname)
        scan['fieldmap']=array(load(fieldmapname).get_data().squeeze())
        #scan['fieldmap'][scan['fieldmap']==-500]=0 #direct nii
        scan['fieldmap']=fliplr(rot90(fliplr(scan['fieldmap']))) #mipav nii
        #scan['fieldmap']=fliplr(rot90(scan['fieldmap']))  #direct nii
        #scan['fieldmap']/=max(abs(scan['fieldmap'].flatten()))
        #scan['fieldmap']=flipud(scan['fieldmap']) #not sure..
        
        #FIELD MAP SCALING!
        #test=parrec.load('/home/eric/data/20120802_prop_testing2/20120802_ETP_prop_test2_WIP_B0MAP_34_1.PAR')
        #header=test.get_header()
        #imdefs=header.image_defs
        #intercept=imdefs[1][11]
        #slope=imdefs[1][12]
    scan['scaninfo']['fieldmap_filename']=fieldmapname
    
    #and read any SENSE sensitivity maps that were acquired
    scan['scaninfo']['sensemap_filename']=sensefilename
    if sensefilename!='':
        print('------------reading SENSE file-----------------------------------')
        tmp=read_philips_list(sensefilename)
        imgs=philips_dic_compactify(tmp)
        del tmp
        si=shape(imgs)
        #print si
        ncoils=si[-1]-1
        nslices=scan['scaninfo']['nslices']
        thresh=max(imgs[:,:,:,ncoils].ravel())/5 #arbitrary threshold
        mask=zeros(si[:-1],'bool')
        mask[imgs[:,:,:,ncoils]>thresh]=1
        for nc in range(ncoils):
            imgs[:,:,:,nc]/=imgs[:,:,:,ncoils]
            imgs[:,:,:,nc]*=mask
        imgs=imgs[:,:,:,0:ncoils].transpose([2,0,1,3])
        #imgs=imgs[:,::-1,:,:] #flip horizontal in the final image
        #imgs=imgs[:,:,::-1,:] #flip image in z
        si=shape(imgs)
        fovxy=coords[0] #todo, make getting the slice location automatic!
        fovz=coords[1]
        zloc=coords[2]
        
        fov_sense=[300.0,300.0,450.0] #guess for now in x,y,z
        rescalesize=max(scan['scaninfo']['xsize'],scan['scaninfo']['ysize'])/2.0
        pixdim=[fovxy/rescalesize,fovxy/rescalesize,fovz/nslices]
        pixdim_sense=[fov_sense[0]/si[0],fov_sense[1]/si[1],fov_sense[2]/si[2]]
        #sp=[rescalesize,rescalesize,nslices]
        #print pixdim
        #print pixdim_sense
        #print si
        #print sp
        #print fov_sense
        vec=[pixdim[0]/pixdim_sense[0],pixdim[1]/pixdim_sense[1],pixdim[2]/pixdim_sense[2],1,1]
        sp_fiction=array(si[0:3])/array(vec[0:3])
        #print(fov_sense,fovxy)
        #print(si[0:3],vec[0:3],sp_fiction)
        #print sp_fiction
        
        transmat1=eye(5)  #[rescalesize/2,rescalesize/2,si[2]/2+fov_sense[2]*zloc/si[2],0]
        transmat2=eye(5)
        #u/d, l/r, a/p
        #NOTE: this is not quite right, I am still missing something in here! What could it be?
        #I am doing a shift for FOV and a shift for the image size...
        #transmat1[:,4]=[si[0]*pixdim_sense[0]/(pixdim[0]*2.0),si[1]/2.0,si[2]*pixdim_sense[2]/(2.0*pixdim[2]),0,1]#zloc/fov_sense[2]+si[2]/2,0,1]
        transmat1[:,4]=[sp_fiction[0]/2.0,sp_fiction[1]/2.0,sp_fiction[2]/2.0,0,1]
        #the transmat1 below is almost right using a transmat2 of eye!
        #transmat1[:,4]=[-rescalesize/2.0*fovxy/fov_sense[0]+rescalesize/2.0,-rescalesize/2.0*fovxy/fov_sense[1]+rescalesize/2.0,pixdim[2]*si[2]/2.0,0,1]
        #vec=[fovxy*si[0]/(rescalesize*fov_sense[0]),fovxy*si[1]/(rescalesize*fov_sense[1]),fovz*si[2]/fov_sense[2],1,1]
        
        affinemat=diag(vec)
        
        #transmat2[:,4]=[pixdim_sense[0]*si[0]/2.0,pixdim_sense[1]*si[1]/2.0,pixdim_sense[2]*si[2]/2,0,1]
        #transmat2[:,4]=[fov_sense[0]*pixdim[0]/(pixdim_sense[0]*2.0),fov_sense[1]*pixdim[1]/(pixdim_sense[1]*2.0),fov_sense[2]*pixdim[2]/(2.0),0,1]
        #transmat1 and transmat2 make a good z combo!
        #transmat2[:,4]=[sp[0]/2.0,sp[1]/2.0,si[2]*pixdim_sense[2]/(2.0*pixdim[2]),0,1]
        #transmat2[:,4]=[sp[0]*pixdim[0]/(2.0*pixdim_sense[0]),sp[1]*pixdim[1]/(2.0*pixdim_sense[1]),si[2]*pixdim_sense[2]/(2.0*pixdim[2]),0,1]
        transmat2[:,4]=[-si[0]/2.0,-si[1]/2.0,-si[2]/2,0,1]#*pixdim_sense[2]/(2.0*pixdim[2]),0,1]
        #transmat2[:,4]=[sp_fiction[0]/2.0,sp_fiction[1]/2.0,sp_fiction[2]/2.0,0,1]
        tmat=dot(dot(transmat1,affinemat),transmat2)
        #translation=tmat[:-1,4]
        #translation[2]=150.0
        translation=[(fov_sense[0]-fovxy)/2.0,(fov_sense[1]-fovxy)/2.0,zloc+sp_fiction[2]/2.0,0]
        vector=diag(tmat)[:-1]
        #print transmat1
        #print affinemat
        #print transmat2
        #print tmat
        #print transmat1
        #print affinemat
        #print transmat2
        #print dot(transmat1,affinemat)
        #print inv(transmat2)
        #print vector
        #print translation
        #scan['datainfo']['sense']=zeros([rescalesize,rescalesize,])
        #scan['datainfo']['sense_orig']=imgs
        scan['datainfo']['sense']=affine_transform(imgs.real,vector,output_shape=[rescalesize,rescalesize,nslices,si[3]],offset=translation)+1j*affine_transform(imgs.imag,vector,output_shape=[rescalesize,rescalesize,nslices,si[3]],offset=translation)
        scan['datainfo']['sense']=scan['datainfo']['sense']/max(abs(scan['datainfo']['sense'].ravel())) #normalize to 1
        #scan['datainfo']['sense']=affine_transform(imgs.real,[fovxy*si[0]/(rescalesize*fov_sense[0]),fovxy*si[1]/(rescalesize*fov_sense[1]),fovz*si[2]/fov_sense[2],1],output_shape=[rescalesize,rescalesize,nslices,si[3]],offset=[rescalesize/2,rescalesize/2,si[2]/2+fov_sense[2]*zloc/si[2],0])+1j*affine_transform(imgs.imag,[fovxy*si[0]/(rescalesize*fov_sense[0]),fovxy*si[1]/(rescalesize*fov_sense[1]),fovz*si[2]/fov_sense[2],1],output_shape=[rescalesize,rescalesize,nslices,si[3]],offset=[rescalesize/2,rescalesize/2,si[2]/2+fov_sense[2]*zloc/si[2],0])
        scan['datainfo']['sense_orig']=scan['datainfo']['sense'].copy()
        
        if False:
            npix=rescalesize**2
            xlocs,ylocs=meshgrid(linspace(-rescalesize/2,rescalesize/2,rescalesize),linspace(-rescalesize/2,rescalesize/2,rescalesize))
            xlocs=xlocs.ravel()
            ylocs=ylocs.ravel()
            print(shape(xlocs))
            v_full = array([ones(npix), xlocs, ylocs, xlocs**2, xlocs*ylocs, ylocs**2])#, xlocs**3, ylocs*xlocs**2, xlocs*ylocs**2, ylocs**3])
            for k in range(ncoils):
                locs=abs(scan['datainfo']['sense'][:,:,0,k].ravel())>0.01
                ct=len(xlocs[locs])
                print(shape(xlocs[locs]))
                print(ct)
                v = array([ones(ct), xlocs[locs], ylocs[locs], xlocs[locs]**2, xlocs[locs]*ylocs[locs], ylocs[locs]**2])#, xlocs[locs]**3, ylocs[locs]*xlocs[locs]**2, xlocs[locs]*ylocs[locs]**2, ylocs[locs]**3])
                print(shape(scan['datainfo']['sense'][:,:,0,k]))
                print(shape(v),shape(v_full))
                imdata=scan['datainfo']['sense'][:,:,0,k].ravel()
                print(shape(imdata[locs]))
                m,_,_,_=lstsq(v.T,imdata[locs]) #fix 0 slices
                print(shape(m))
                #imdata[~locs]=0
                #scan['datainfo']['sense'][:,:,0,k]=imdata.reshape([rescalesize,rescalesize])#dot(m,v_full).reshape([rescalesize,rescalesize])
                scan['datainfo']['sense'][:,:,0,k]=dot(m,v_full).reshape([rescalesize,rescalesize])
                
        
        
        #old k-space approach
#        scan['datainfo']['sense']=read_philips_list(sensefilename)
#        #sum averages
#        scan['datainfo']['sense']['image']={}
#        for elem in scan['datainfo']['sense']['kspace']:
#            try:
#                scan['datainfo']['sense']['image'][elem[0]]=scan['datainfo']['sense']['kspace'][elem]
#            except:
#                scan['datainfo']['sense']['image'][elem[0]]+=scan['datainfo']['sense']['kspace'][elem]
#        s=shape(scan['datainfo']['sense']['image'][0])
#        center=s[1]/2
#        dist=s[0]/2
#        #check the spatial registration!
#        for elem in scan['datainfo']['sense']['image']:
#            #not sure about the extra ifftshift on the following line
#            scan['datainfo']['sense']['image'][elem]=ifftshift(ifftshift(ifftn(ifftshift(scan['datainfo']['sense']['image'][elem]))),axes=0)
#            scan['datainfo']['sense']['image'][elem]=rot90(scan['datainfo']['sense']['image'][elem][:,center-dist:center+dist,:],-1)
#            scan['datainfo']['sense']['image'][elem]=medfilt(scan['datainfo']['sense']['image'][elem][:,:,0].real,7)+1j*medfilt(scan['datainfo']['sense']['image'][elem][:,:,0].imag,7)
        print('------------done reading SENSE file------------------------------')
                
    return scan
    


def read_philips_list_raw(scan): #OUTDATED
    from numpy import zeros, prod
    from struct import unpack
    
    xsize=scan['scaninfo']['xsize']
    ysize=scan['scaninfo']['ysize']
    zsize=scan['scaninfo']['zsize']
    if scan['scaninfo']['ifrefscan']: #if it is a refscan
        ncoils=scan['scaninfo']['ncoils']+1
        nslices=scan['scaninfo']['nslices']-1
    else:
        ncoils=scan['scaninfo']['ncoils']
        nslices=scan['scaninfo']['nslices']
    ndynamics=scan['scaninfo']['ndynamics']
    naverages=scan['scaninfo']['naverages']
    nextra1=scan['scaninfo']['nextra1']
    nextra2=scan['scaninfo']['nextra2']
    matrix=[ysize,xsize,zsize,ncoils,nslices,ndynamics,naverages,nextra1,nextra2] #zsize is untested
    scan['scaninfo']['ksize']=matrix
    #OK, this is bumping up on memory issues. I should consider just doing a
    #single image allocation for the images and coils, and use dictionary keys
    #to encode everything beyond that.
    print ['Allocating a',prod(matrix)*8/(1024*1024),'MB image:',matrix]
    kspace=zeros(matrix,complex)
    f=open(scan['datafilename'],'rb')
    for k in scan['datainfo']['STD']:
        if ncoils==1: #it seems uninitialized
            k[5]=0
        if scan['scaninfo']['ifrefscan']==1: #seems like unique numbers for the refscan
            y_tmp=k[8] #using image coordinates, not kspace coordinates
            if k[4]==1: #body coil I believe
                k[4]=0 #set location to 0
                k[5]=0 #set coil to 0
            else:
                k[5]+=1 #if a multi-channel coil
        else:
            y_tmp=k[8]+scan['scaninfo']['ky_range'][1] #kspace coordinates
        f.seek(k[19])
        #print('{}f'.format(k[18]/4))
        floats=unpack('{}f'.format(k[18]/4),f.read(k[18]))
        #print flen
        #print k[12]
        for fl in range(matrix[1]): #always even
            #if k[12]==1 or k[12]==0: #it looks like the sign variable indicates the direction, but the reversal has already occurred
            #    fl_tmp=fl
            #else:
            #    fl_tmp=matrix[1]-fl-1
            kspace[y_tmp,fl,k[9],k[5],k[4],k[1],k[11],k[6],k[7]]=complex(floats[2*fl],floats[2*fl+1])
            #kspace[y_tmp,fl,k[9],k[5],k[4],k[1],k[11],k[6],k[7]]=1
            
            
        #print k
    #print scan
    scan['kspace']=kspace
    return scan
    
    
def read_philips_dic_raw(scan): #reads the images into the dictionary rather than as a huge image
    from numpy import zeros, prod, array#, roll, conj
    from struct import unpack
    
    scan['kspace']={}
    scan['phasecorr_orig']={}
    scan['frequencycorr_orig']={}
    scan['scaninfo']['readout_direction']={}
    #scan['kspace'][0]={}
    #scan['kspace'][0][0]={}
    #scan['kspace'][0][0][0]={}
    #scan['kspace'][0,0,0,0,0,0]={}
    #scan['kspace'][5][0]={}
    #scan['kspace'][0][0][0][0][0]=[]
    
    xsize=scan['scaninfo']['xsize']
    ysize=scan['scaninfo']['ysize']
    zsize=scan['scaninfo']['zsize']
    #if scan['scaninfo']['ifrefscan']: #if it is a refscan
        #ncoils=scan['scaninfo']['ncoils']+1
        #nslices=scan['scaninfo']['nslices']-1
    #else:
        #ncoils=scan['scaninfo']['ncoils']
        #nslices=scan['scaninfo']['nslices']
    #ndynamics=scan['scaninfo']['ndynamics']
    #naverages=scan['scaninfo']['naverages']
    #nextra1=scan['scaninfo']['nextra1']
    #nextra2=scan['scaninfo']['nextra2']
    matrix=[ysize,xsize,zsize] #ncoils,nslices,ndynamics,naverages,nextra1,nextra2] #zsize is untested
    scan['scaninfo']['ksize']=matrix
    #OK, this is bumping up on memory issues. I should consider just doing a
    #single image allocation for the images and coils, and use dictionary keys
    #to encode everything beyond that.
    print ['The maximum size is',[ysize,xsize,zsize,scan['scaninfo']['ncoils'],scan['scaninfo']['nslices'],scan['scaninfo']['ndynamics'],scan['scaninfo']['naverages'],scan['scaninfo']['nextra1'],scan['scaninfo']['nextra2']]]
    nimages=0
    #kspace=zeros(matrix,complex)
    f=open(scan['datafilename'],'rb')
    ctr=0
    for k in scan['datainfo']['STD']:
        #if ncoils==1: #it seems uninitialized
            #k[5]=0
        if scan['scaninfo']['ifrefscan']==1: #seems like unique numbers for the refscan
            y_tmp=k[8] #using image coordinates, not kspace coordinates
            #if k[4]==1: #body coil I believe
                #k[4]=0 #set location to 0
                #k[5]=0 #set coil to 0
            #else:
                #k[5]+=1 #if a multi-channel coil
        else:
            y_tmp=k[8]+scan['scaninfo']['ky_range'][1] #kspace coordinates
            #print y_tmp
        #print [k[4],k[1],k[11],k[6],k[7]]
        try: #we need to create the image if it doesn't exist
            scan['kspace'][k[5],k[4],k[1],k[11],k[6],k[7]][0,0,0]
        except KeyError:
            #print ['KeyError',k[4],k[1],k[11],k[6],k[7]]
            scan['kspace'][k[5],k[4],k[1],k[11],k[6],k[7]]=zeros(matrix,complex)
            nimages+=1
            scan['scaninfo']['readout_direction'][k[5],k[4],k[1],k[11],k[6],k[7]]=zeros([ysize,zsize],int)
        #print k
        scan['scaninfo']['readout_direction'][k[5],k[4],k[1],k[11],k[6],k[7]][y_tmp,k[9]]=int(k[12])
        #print type(k[19])
        if scan['scaninfo']['ifcpx']==1:
            if y_tmp==0:
                ctr+=1
            f.seek(k[19]+512*ctr) #for some reason cpx data has 512 byte buffers at each image
        else:
            f.seek(k[19])
        
        #print('{}f'.format(k[18]/4))
        floats=array(unpack('{}f'.format(k[18]/4),f.read(k[18])))
        #print flen
        #print k[12]
        
        scan['kspace'][k[5],k[4],k[1],k[11],k[6],k[7]][y_tmp,:,k[9]]=floats[::2]+1j*floats[1::2]
        #for fl in range(matrix[1]): #always even
            #the phase correction takes the readout shifting into account. I am not sure how I feel about that, but thats how it is.
            #if k[12]==1 or k[12]==0: #it looks like the sign variable indicates the direction, but the reversal has already occurred
            #fl_tmp=fl
            #else:
            #    fl_tmp=matrix[1]-fl-1
            #if fl==0:
            #    scan['kspace'][k[5],k[4],k[1],k[11],k[6],k[7]][y_tmp,fl,k[9]]=complex(1e6,0)
            #else:
            #scan['kspace'][k[5],k[4],k[1],k[11],k[6],k[7]][y_tmp,fl,k[9]]=complex(floats[2*fl],floats[2*fl+1]) #IS GOOD WITH LOOP
            #scan['kspace'][k[5],k[4],k[1],k[11],k[6],k[7]][y_tmp,fl,k[9]]=1
            #[channel,location,dynamics,average,extra1,extra2]
     
    #There is an issue with the centering of the FFT which I should deal with somehow... 
    #With odd numbers of phase encodes at least.
    #for element in scan['kspace']:
    #    scan['kspace'][element]=roll(scan['kspace'][element],1,axis=0)
    #    print scan['kspace'][element][0,15,0]
    #    scan['kspace'][element][0,:,0]=conj(scan['kspace'][element][0,:,0])
    #    print scan['kspace'][element][0,15,0]
        
    scan['scaninfo']['nimages']=nimages
        
    if scan['scaninfo']['ifrefscan']==0:
        for k in scan['datainfo']['PHX']:
            scan['phasecorr_orig'][k[8],k[9],k[5],k[4],k[12]]=zeros(scan['scaninfo']['X-resolution'],complex)
            f.seek(k[19])
            floats=unpack('{}f'.format(k[18]/4),f.read(k[18]))
            for fl in range(scan['scaninfo']['X-resolution']): #always even
                #if k[12]==1 or k[12]==0: #it looks like the sign variable indicates the direction, but the reversal has already occurred
                #fl_tmp=fl
                #else:
                #    fl_tmp=matrix[1]-fl-1
                scan['phasecorr_orig'][k[8],k[9],k[5],k[4],k[12]][fl]=complex(floats[2*fl],floats[2*fl+1])
                #xsize,zsize,coil,location,sign
                
        for k in scan['datainfo']['FRX']:
            scan['frequencycorr_orig'][k[8],k[9],k[5],k[4],k[12]]=zeros(scan['scaninfo']['X-resolution'],complex)
            f.seek(k[19])
            floats=unpack('{}f'.format(k[18]/4),f.read(k[18]))
            for fl in range(scan['scaninfo']['X-resolution']): #always even
                #if k[12]==1 or k[12]==0: #it looks like the sign variable indicates the direction, but the reversal has already occurred
                #fl_tmp=fl
                #else:
                #    fl_tmp=matrix[1]-fl-1
                scan['frequencycorr_orig'][k[8],k[9],k[5],k[4],k[12]][fl]=complex(floats[2*fl],floats[2*fl+1])
            
    f.close()
        #print k
    #print scan
    #scan['kspace']=kspace
    print ['There is/are',nimages,'image(s)']
    print ['Total image size is',nimages*prod(matrix)*8/(1024*1024),'MB']
    return scan
    
    
def read_philips_dic_fiddle(scan):
    from numpy import zeros, unique
    #This function just reshapes and fiddles with the images from read_philips_dic_raw to get a reasonable image
    #the format is: [ncoils, nslices,ndynamics,naverages,nextra1,nextra2][y,x,z]
    
    #first, we need to figure out the correct size of the image
    #The naverages, nextra1, and nextra2 all lie if the image is diffusion, and for the refscan, ncoils also lies
    
    xsize=scan['scaninfo']['xsize']
    ysize=scan['scaninfo']['ysize']
    zsize=scan['scaninfo']['zsize']
    #if scan['scaninfo']['ifrefscan']: #if it is a refscan
    #    ncoils=scan['scaninfo']['ncoils']+1
    #    nslices=scan['scaninfo']['nslices']-1
    #else:
    #    ncoils=scan['scaninfo']['ncoils']
    #    nslices=scan['scaninfo']['nslices']
    #ndynamics=scan['scaninfo']['ndynamics']
    #naverages=scan['scaninfo']['naverages'] #averages are just averages (it looks like there are 2 b=0 images by default)
    #nextra1=scan['scaninfo']['nextra1'] #this looks like the number of diffusion values (including b=0)
    #nextra2=scan['scaninfo']['nextra2'] #this looks like the number of diffusion directions (NOT including b=0)
    #if nextra2>1: #it is a diffusion scan
    #    ndirs=nextra1
    #    naverages-=1 #but what if we have averages?
    #else:
    #    ndirs=1
    
    coils=[]
    slices=[]
    dynamics=[]
    averages=[]
    extra1=[]
    extra2=[]
    for element in scan['kspace']:
        coils.append(element[0])
        slices.append(element[1])
        dynamics.append(element[2])
        averages.append(element[3])
        extra1.append(element[4])
        extra2.append(element[5])
    coils=unique(coils)
    slices=unique(slices)
    dynamics=unique(dynamics)
    averages=unique(averages)
    extra1=unique(extra1)
    extra2=unique(extra2)
    ncoils=len(coils)
    nslices=len(slices)
    ndynamics=len(dynamics)
    naverages=len(averages)
    #nextra1=len(extra1)
    #nextra2=len(extra2)
    ndirs=len(extra1)
    print ['coils',coils]
    print ['slices',slices]
    print ['dynamics',dynamics]
    print ['averages',averages]
    print ['extra1',extra1]
    print ['extra2',extra2]
    
    matrix=[ysize,xsize,zsize,ncoils,nslices,ndynamics,naverages,ndirs]
    kspace=zeros(matrix,complex)
    for element in scan['kspace']:
        kspace[:,:,:,element[0],element[1],element[2],element[3],element[4],element[5]]=scan['kspace'][element]
        
    return scan
    
    
    
def philips_dic_compactify(scan,image='kspace'):
    from numpy import zeros, shape
    #note, this now orders the output images
    
    channel=set([])
    location=set([])
    dynamics=set([])
    average=set([])
    extra1=set([])
    extra2=set([])
    for element in scan[image]:
        channel.add(element[0])
        location.add(element[1])
        dynamics.add(element[2])
        average.add(element[3])
        extra1.add(element[4])
        extra2.add(element[5])
    channel=list(channel)
    location=list(location)
    dynamics=list(dynamics)
    average=list(average)
    extra1=list(extra1)
    extra2=list(extra2)
    
    if image=='kspace':
        kspace=zeros([scan['scaninfo']['ysize'],scan['scaninfo']['xsize'],scan['scaninfo']['zsize'],scan['scaninfo']['nimages']],complex)
    else:
        s=shape(scan['data'][0,0,0,0,0,0])
        k=0
        for elem in scan['data']:
            k+=1
        kspace=zeros([s[0],s[1],s[2],k],complex)
        
    k=0
    
    for loc in location:
        for dyn in dynamics:
            for ext1 in extra1:
                for ext2 in extra2:
                    for chan in channel:
                        for avg in average:
                            try:
                                kspace[:,:,:,k]=scan[image][chan,loc,dyn,avg,ext1,ext2]
                                k+=1
                            except:
                                pass
                            
    #for element in scan['kspace']:
    #    kspace[:,:,:,k]=scan['kspace'][element]
    #    k+=1
    
    return kspace #this is a little different than the other functions!
    
    
def philips_dic_compactify_image(scan,imname='img'):
    from numpy import zeros, shape
    #note, this now orders the output images
    #print scan['scaninfo']['imsize']
    #print imname
    if imname=='img':
        sizename='imsize'
        nims='nimages'
        #elif imname=='img_square':
    else:
        sizename='imsize_square'
        nims='recon_nimages'
    channel=set([])
    location=set([])
    dynamics=set([])
    average=set([])
    extra1=set([])
    extra2=set([])
    for element in scan[imname]:
        channel.add(element[0])
        location.add(element[1])
        dynamics.add(element[2])
        average.add(element[3])
        extra1.add(element[4])
        extra2.add(element[5])
    channel=list(channel)
    location=list(location)
    dynamics=list(dynamics)
    average=list(average)
    extra1=list(extra1)
    extra2=list(extra2)
    
    #print('channel=',channel)
    #print('loaction=',location)
    #print('dynamics=',dynamics)
    #print('average=',average)
    #print('extra1=',extra1)
    #print('extra2=',extra2)
    
    #print sizename
    img=zeros([scan['scaninfo'][sizename][0],scan['scaninfo'][sizename][1],scan['scaninfo']['nslices'],scan['scaninfo'][nims]/scan['scaninfo']['nslices']],complex)
    #print shape(img)
    k=0
    
    
    for dyn in dynamics:
        for ext1 in extra1:
            for ext2 in extra2:
                for chan in channel:
                    for avg in average:
                        for loc in location:
                            try:
                                img[:,:,loc,k]=scan[imname][chan,loc,dyn,avg,ext1,ext2][:,:,0] #2D only!
                                #k+=1
                            except:
                                #print [chan,loc,dyn,avg,ext1,ext2]
                                pass
                                
                        try:
                            scan[imname][chan,0,dyn,avg,ext1,ext2]
                            k+=1 #stupid counter....
                            #print k
                        except:
                            pass
    #print k
    #for element in scan[imname]:
        #print shape(scan['img'][element])
    #    print element
    #    img[:,:,:,k]=scan[imname][element]
    #    k+=1
    
    return img #this is a little different than the other functions!
    
def phase_correct_philips(scan,phasecorr=1):
    from numpy import zeros, interp, linspace, exp, pi, linspace, ones, conj, sort, array, nonzero, shape, meshgrid, tile, expand_dims, atleast_3d
    from numpy.fft import fftshift, ifftshift, ifft, ifftn, fftn, fft
    from copy import copy
    from scipy.ndimage.filters import gaussian_filter1d
    
    scan['data_tmp']={}
    scan['data']={}
    scan['phasecorr']={}
    scan['frequencycorr']={}
    scan['img']={}
    scan['img_square']={}
    scan['img_square_cg']={}
    
    imsize=copy(scan['scaninfo']['ksize'])
    imsize[1]=int(imsize[1]/scan['scaninfo']['kx_oversample_factor'])
    scan['scaninfo']['imsize']=copy(imsize)
    #imsize=scan['scaninfo']['imsize']
    cropidx=[scan['scaninfo']['ksize'][1]/2-imsize[1]/2,scan['scaninfo']['ksize'][1]/2+imsize[1]/2]
    freqcorr_warning=1
    if phasecorr==1: #I need to acquire correction data for each dynamic angle!!!
        x=linspace(scan['scaninfo']['X_range'][0],scan['scaninfo']['X_range'][1],scan['scaninfo']['imsize'][1])
        xp=linspace(scan['scaninfo']['X_range'][0],scan['scaninfo']['X_range'][1],scan['scaninfo']['X_range'][1]-scan['scaninfo']['X_range'][0]+1)
        if scan['scaninfo']['auxiliary_phase_data_filename']!='':
            #if we are using externally collected phase data
            #we are just going to go ahead and assume that the data size is correct
            
            #note that correction_angles is only used for interpolation
            #if scan['scaninfo']['ndynamics']==scan['datainfo']['auxiliary_phase_data']['scaninfo']['ndynamics']:
            correction_angles=180*array(range(scan['datainfo']['auxiliary_phase_data']['scaninfo']['ndynamics']))/float(scan['datainfo']['auxiliary_phase_data']['scaninfo']['ndynamics'])
            #print correction_angles
            for elem in scan['datainfo']['auxiliary_phase_data']['kspace']:
                #print elem
                #angle=180*elem[2]/float(scan['scaninfo']['ndynamics'])
                #dists=abs(correction_angles-angle)
                #dists_sort=sort(dists)[:2]
                #idxs=nonzero(dists<=dists_sort[1])[0][:2]
                #dists_use=dists[idxs]
                #print('angle=',angle)
                #print('dists=',dists)
                #print('dists_sort=',dists_sort)
                ##print sort(dists)
                #print('idxs=',idxs)
                #print('dists_use=',dists_use)
                #dists_use/=sum(dists_use)
                #dists_use=-1*(dists_use-1)
                #print('dists_use=',dists_use)
                #tmp1=fftshift(ifft(fftshift(scan['datainfo']['auxiliary_phase_data']['kspace'][elem[0],elem[1],idxs[0],0,0,0],1),None,1),1)
                #tmp1=gaussian_filter1d(tmp1.real,5,1)+1j*gaussian_filter1d(tmp1.imag,5,1)
                #tmp2=fftshift(ifft(fftshift(scan['datainfo']['auxiliary_phase_data']['kspace'][elem[0],elem[1],idxs[1],0,0,0],1),None,1),1)
                #tmp2=gaussian_filter1d(tmp2.real,5,1)+1j*gaussian_filter1d(tmp2.imag,5,1)
                #tmp=dists_use[0]*tmp1+dists_use[1]*tmp2
                ##tmp=tmp[:,cropidx[0]:cropidx[1],:]
                ##scan['phasecorr'][k,kk,elem[0],elem[1],dyn]
                ##tmp[tmp==0]=1 #try to get rid of zeros in tmp
                ##print tmp
                #for k in range(scan['scaninfo']['imsize'][0]):
                #    for kk in range(scan['scaninfo']['imsize'][2]):
                #        scan['phasecorr'][k,kk,elem[0],elem[1],elem[2]]=conj(tmp[k,:,kk]/abs(tmp[k,:,kk]))
                try: #if it exists already, just don't run it again.
                    scan['phasecorr'][0,0,elem[0],elem[1],0]
                    #print 'phasecorr exists!'
                except:
                    for dyn in range(scan['scaninfo']['ndynamics']):
                        #print scan['datainfo']['auxiliary_phase_data']['kspace']
                        
                        #if scan['scaninfo']['ndynamics']==scan['datainfo']['auxiliary_phase_data']['scaninfo']['ndynamics']:
                        #    tmp=fftshift(ifft(fftshift(scan['datainfo']['auxiliary_phase_data']['kspace'][elem],0),None,0),0)
                        #    tmp=tmp[:,cropidx[0]:cropidx[1],:]
                        #else: #if we don't have the same number of external phase angles as acquired angles
                            
                        #print elem
                        angle=180*dyn/float(scan['scaninfo']['ndynamics'])
                        dists=abs(correction_angles-angle)
                        dists_sort=sort(dists)[:2]
                        idxs=nonzero(dists<=dists_sort[1])[0][:2]
                        dists_use=dists[idxs]
                        #print('angle=',angle)
                        #print('dists=',dists)
                        #print('dists_sort=',dists_sort)
                        #print sort(dists)
                        #print('idxs=',idxs)
                        #print('dists_use=',dists_use)
                        dists_use/=sum(dists_use)
                        dists_use=-1*(dists_use-1)
                        #print('dists_use=',dists_use)
                        tmp1=ifftshift(ifft(ifftshift(scan['datainfo']['auxiliary_phase_data']['kspace'][elem[0],elem[1],idxs[0],0,0,0],1),None,1),1)
                        #tmp1=gaussian_filter1d(tmp1.real,5,1)+1j*gaussian_filter1d(tmp1.imag,5,1)
                        tmp2=ifftshift(ifft(ifftshift(scan['datainfo']['auxiliary_phase_data']['kspace'][elem[0],elem[1],idxs[1],0,0,0],1),None,1),1)
                        #tmp2=gaussian_filter1d(tmp2.real,5,1)+1j*gaussian_filter1d(tmp2.imag,5,1)
                        tmp=dists_use[0]*tmp1+dists_use[1]*tmp2
                        #print shape(tmp)
                        tmp=tmp[:,cropidx[0]:cropidx[1],:]  #crop the readout FOV
                        #print shape(tmp)
                        #scan['phasecorr'][k,kk,elem[0],elem[1],dyn]
                        #tmp[tmp==0]=1 #try to get rid of zeros in tmp
                        #print tmp
                        
                        #phase correction insanity!
                        mid=scan['scaninfo']['imsize'][0]/2-(scan['scaninfo']['imsize'][0]/2)%2
                        #mid=0
                        for k in range(scan['scaninfo']['imsize'][0]):
                            for kk in range(scan['scaninfo']['imsize'][2]):
                                #scan['phasecorr'][k,kk,elem[0],elem[1],dyn]=conj(tmp[k,:,kk])/abs(tmp[k,:,kk])  #was k instead of mid
                                #using the middle lines really helps when using field map corrections!
                                #I think the phase gets modified with the field map as well, so in some ways it gets over-compensated
                                if k%2==0: #I am only using the central correciton data, why is this such an issue?
                                    scan['phasecorr'][k,kk,elem[0],elem[1],dyn]=conj(tmp[mid,:,kk])/abs(tmp[mid,:,kk])  #was k instead of mid
                                else: #TODO: fix this issue!
                                    scan['phasecorr'][k,kk,elem[0],elem[1],dyn]=conj(tmp[mid-1,:,kk])/abs(tmp[mid-1,:,kk])  #was k instead of mid (trying mid-1)
                #        angle=180*dyn/float(scan['scaninfo']['ndynamics'])
                #        dists=abs(correction_angles-angle)
                #        dists_sort=sort(dists)[:2]
                #        idxs=nonzero(dists<=dists_sort[1])[0]
                #        print('angle=',angle)
                #        print('dists=',dists)
                #        print('dists_sort=',dists_sort)
                #        #print sort(dists)
                #        print('idxs=',idxs)
                #        tmp1=fftshift(ifft(fftshift(scan['datainfo']['auxiliary_phase_data']['kspace'][elem[0],elem[1],idxs[0],0,0,0],1),None,1),1)
                #        tmp1=gaussian_filter1d(tmp1.real,5,1)+1j*gaussian_filter1d(tmp1.imag,5,1)
                #        tmp2=fftshift(ifft(fftshift(scan['datainfo']['auxiliary_phase_data']['kspace'][elem[0],elem[1],idxs[1],0,0,0],1),None,1),1)
                #        tmp2=gaussian_filter1d(tmp2.real,5,1)+1j*gaussian_filter1d(tmp2.imag,5,1)
                #        tmp=(dists_sort[0]*tmp1+dists_sort[1]*tmp2)/sum(dists_sort)
                #        tmp=tmp[:,cropidx[0]:cropidx[1],:]
                #        #scan['phasecorr'][k,kk,elem[0],elem[1],dyn]
                #        #tmp[tmp==0]=1 #try to get rid of zeros in tmp
                #        #print tmp
                #        for k in range(scan['scaninfo']['imsize'][0]):
                #            for kk in range(scan['scaninfo']['imsize'][2]):
                #                scan['phasecorr'][k,kk,elem[0],elem[1],dyn]=conj(tmp[k,:,kk]/abs(tmp[k,:,kk]))
                        #print [0,0,elem[0],elem[1],dyn]
                        
            for elem in scan['frequencycorr_orig']: #lets just assume we have to rescale the frequency!
                #print elem
                scan['frequencycorr'][elem]=interp(x,xp,scan['frequencycorr_orig'][elem].real) #it is real only
                freqcorr_warning=0
        else: #if we are using internally collected phase data
            if scan['scaninfo']['imsize']!=(scan['scaninfo']['X_range'][1]-scan['scaninfo']['X_range'][0]+1):
                print 'warning, interpolating phase correction!'
                for elem in scan['phasecorr_orig']:
                    #print elem
                    #scan['phasecorr'][0,elem[1],elem[2],elem[3],elem[4]]=exp(-1j*interp(x,xp,angle(scan['phasecorr_orig'][elem])))
                    scan['phasecorr'][0,elem[1],elem[2],elem[3],elem[4]]=interp(x,xp,scan['phasecorr_orig'][elem].real)-1j*interp(x,xp,scan['phasecorr_orig'][elem].imag)
                    #print max(abs(scan['phasecorr'][elem]))
                    #print min(abs(scan['phasecorr'][elem]))
                    #plot(angle(scan['phasecorr'][elem]))
                    #plot(angle(scan['phasecorr_orig'][elem]))
                    #show()
                for elem in scan['frequencycorr_orig']:
                    scan['frequencycorr'][elem]=interp(x,xp,scan['frequencycorr_orig'][elem].real) #it is real only
                    freqcorr_warning=0
                    #scan['phasecorr'][0,elem[1],elem[2],elem[3],elem[4]]=interp(x,xp,scan['phasecorr_orig'][elem].real)+1j*interp(x,xp,angle(scan['phasecorr_orig'][elem].imag))
            else:
                for elem in scan['phasecorr_orig']:
                    scan['phasecorr'][0,elem[1],elem[2],elem[3],elem[4]]=scan['phasecorr_orig'][elem]
                    scan['frequencycorr'][elem]=scan['frequencycorr_orig'][elem]

    if freqcorr_warning==1: #why is this even happening?
        print('No frequency correction data available!')
        
        
    maxmatrix=max(imsize[0],imsize[1])    
    scan['scaninfo']['imsize_square']=[maxmatrix,maxmatrix,imsize[2]]
    #scan['scaninfo']['imsize_square']=[int(imsize[0]*float(imsize[1])/float(imsize[0])),imsize[1],imsize[2]]
    #print scan['scaninfo']['imsize_square']
    scan['scaninfo']['recon_nimages']=0
    
    #for kspace addition
    
    #offset=[scan['scaninfo']['imsize_square'][0]/2-scan['scaninfo']['imsize'][0]/2,scan['scaninfo']['imsize_square'][0]/2-scan['scaninfo']['imsize'][0]/2+scan['scaninfo']['imsize'][0]]
    #print offset
    
    #offset[:]=offset[:]+1
    #print offset
    #offset[1]=offset[0]+scan['scaninfo']['imsize'][0]
    #blade[:,offset[0]:offset[1]]=1
    
    
    #scan['blade']=blade
    scan['dcomp_tmp']={}
    
    #blade phase correction (NOT EPI phase correction!!!)
    x,y=meshgrid(linspace(-1,1,scan['scaninfo']['imsize'][1],False),linspace(-1,1,scan['scaninfo']['imsize'][2],False))
    #w_consistency=max(abs(y.flatten()))-abs(y.transpose().copy())
    dist=pow(pow(x,2)+pow(y,2),0.5)
    dist=max(dist.flatten())-dist
    dist=tile(atleast_3d(dist),[1,1,scan['scaninfo']['imsize'][2]])
    
    for element in scan['kspace']:
        #for k in range(scan['scaninfo']['ksize'][0]):
        #    for kk in range(scan['scaninfo']['ksize'][2]):
        #        scan['kspace'][element][k,:,kk]=scan['kspace'][element][k,:,kk]*exp(-1j*linspace(-2*pi,2*pi,scan['scaninfo']['ksize'][1],False))
        tmp=ifftshift(ifft(ifftshift(scan['kspace'][element],1),None,1),1)
        scan['img'][element]=copy(tmp[:,cropidx[0]:cropidx[1],:])
        #print shape(scan['img'][element])
        if phasecorr==1:
            for k in range(scan['scaninfo']['imsize'][0]-scan['scaninfo']['ky_range'][1]+scan['scaninfo']['ky_range'][0]-1,scan['scaninfo']['imsize'][0]):
                for kk in range(scan['scaninfo']['imsize'][2]):
                    #print scan['scaninfo']['imsize'][0]-scan['scaninfo']['ky_range'][1]+scan['scaninfo']['ky_range'][0]
                    #print scan['scaninfo']['imsize'][0]
                    #print k
                    #print kk
                    rodir=scan['scaninfo']['readout_direction'][element][k,kk]
                    #rodir=-2*(k%2)+1
                    #rodir=2*(k%2)-1
                    #print type(scan['img'][element][k,:,kk])
                    #print type(scan['phasecorr'][elem[0],0,element[0],element[1],rodir])
                    #print shape(scan['img'][element][k,:,kk]*scan['phasecorr'][elem[0],0,element[0],element[1],rodir])
                    #print [elem[0],0,element[0],element[1],rodir]
                    #print 'phase'
                    #scan['phasecorr'][elem[0],0,element[0],element[1],rodir]
                    #print 'freq'
                    #scan['frequencycorr'][elem[0],0,element[0],element[1],rodir]
                    #for key in scan['frequencycorr']:
                    #    print key
                    #for key in scan['phasecorr']:
                    #    print key
                    #for key in scan['frequencycorr']:
                    #    print key
                    #try:
                    #    scan['frequencycorr'][elem[0],0,element[0],element[1],rodir]
                    #except KeyError:
                    #    print('no frequency correction!')
                    #print k
                    #print elem
                    if scan['scaninfo']['auxiliary_phase_data_filename']!='':
                        #if freqcorr_warning==0:
                        try:
                            #print [k,kk,element[0],element[1],element[2]]
                            #scan['phasecorr'][k,kk,element[0],element[1],element[2]]
                            #scan['frequencycorr'][elem[0],0,element[0],element[1],rodir]
                            scan['img'][element][k,:,kk]=scan['img'][element][k,:,kk]*scan['phasecorr'][k,kk,element[0],element[1],element[2]]/scan['frequencycorr'][elem[0],0,element[0],element[1],rodir]
                        except:
                            scan['img'][element][k,:,kk]=scan['img'][element][k,:,kk]*scan['phasecorr'][k,kk,element[0],element[1],element[2]]
                    else:
                        #if freqcorr_warning==0:
                        try:
                            #print [0,0,element[0],element[1],rodir]
                            
                            #scan['phasecorr'][0,0,element[0],element[1],rodir]
                            #scan['frequencycorr'][elem[0],0,element[0],element[1],rodir]
                            scan['img'][element][k,:,kk]=scan['img'][element][k,:,kk]*scan['phasecorr'][0,0,element[0],element[1],rodir]/scan['frequencycorr'][elem[0],0,element[0],element[1],rodir]
                        except:
                        #print('no frequency correction data available!')
                            scan['img'][element][k,:,kk]=scan['img'][element][k,:,kk]*scan['phasecorr'][0,0,element[0],element[1],rodir]
                    
            #imaging: k[5],k[4],k[1],k[11],k[6],k[7]
            #phase: [k[8],k[9],k[5],k[4],k[12]]

        
        #scan['img'][element]=fftshift(fft(fftshift(scan['img'][element],1),None,1),1)
        
        #scan['img'][element]=fftshift(ifft(fftshift(scan['img'][element],1),None,1),1)
        scan['img'][element]=ifftshift(ifft(ifftshift(scan['img'][element],0),None,0),0)
        
        #scan['img'][element]=fftshift(ifft(fftshift(scan['img'][element],1),None,1),1)
        
        #scan['img'][element]=fftshift(fft(fftshift(scan['img'][element],1),None,1),1)
        
        
        #blade phase correction (NOT EPI phase correction!!!)
        img_phcorr=copy(fftshift(fftn(fftshift(scan['img'][element],axes=(0,1)),axes=(0,1)),axes=(0,1)))
        #print(shape(img_phcorr))
        img_phcorr*=dist
        img_phcorr=ifftshift(ifftn(ifftshift(img_phcorr,axes=(0,1)),axes=(0,1)),axes=(0,1))
        img_phcorr/=abs(img_phcorr)
        scan['img'][element]=scan['img'][element]*img_phcorr.conj()
        
    return scan
    
    
def philips_dic_recon_basic(scan,coilcombinealgorithm='sum',sumdynamics=1,phasecorr=1,sumaverages=1):
    from numpy import zeros, interp, linspace, exp, pi, linspace, ones, conj, sort, array, nonzero
    from numpy.fft import fftshift, ifft, ifftn, fftn
    from copy import copy
    from scipy.ndimage.interpolation import zoom, rotate
    from scipy.ndimage.filters import gaussian_filter1d
    #from nfft_wrappers import init_nfft_2d, init_iterative_nfft_2d, grid_nfft_2d, finalize_nfft_2d, finalize_iterative_nfft_2d
    #from nfft_helpers import propeller_sampling_location_generator, calc_density_weights
    #from matplotlib.pyplot import plot, show
    
    scan['img']={}
    scan['img_square']={}
    scan['phasecorr']={}
    scan['frequencycorr']={}
    scan['scaninfo']['image_counter']={}
    densitycomp={}
    imsize=copy(scan['scaninfo']['ksize'])
    imsize[1]=int(imsize[1]/scan['scaninfo']['kx_oversample_factor'])
    scan['scaninfo']['imsize']=copy(imsize)
    #cropidx=[scan['scaninfo']['ksize'][1]/2-imsize[1]/2,scan['scaninfo']['ksize'][1]/2+imsize[1]/2]
    densctr={}
    
    #print cropidx
    #print imsize
    #print scan['scaninfo']['kx_oversample_factor']
    
    #print shape(x)
    #print shape(xp)
    #print scan['scaninfo']['X_range']
    
    scan=phase_correct_philips(scan,phasecorr)
    offset=[scan['scaninfo']['imsize_square'][0]/2+scan['scaninfo']['ky_range'][0],scan['scaninfo']['imsize_square'][0]/2+scan['scaninfo']['ky_range'][1]]
    blade=zeros(scan['scaninfo']['imsize_square'][0:2])
    blade[offset[0]:offset[1],:]=1
    
    for element in scan['kspace']:
    
        
    
        #print shape(scan['img'][element])
        #zoom(scan['img'][element].imag,[float(imsize[1])/float(imsize[0]),1,1])
        #print element
        #print element_square
        #
        element_square=element
        if sumdynamics==1:
            element_square=(element_square[0],element_square[1],0,element_square[3],element_square[4],element_square[5])
        #else:
            #element_square=element
        #print coilcombinealgorithm
        #if coilcombinealgorithm=='none':
            #print 'coilcombine none!'
            #really just a null operation!!
            #element_square=(element_square[0],element_square[1],element_square[2],element_square[3],element_square[4],element_square[5])
        if coilcombinealgorithm!='none':
            element_square=(0,element_square[1],element_square[2],element_square[3],element_square[4],element_square[5])
        if sumaverages==1:
            element_square=(element_square[0],element_square[1],element_square[2],0,element_square[4],element_square[5])
        
            
        #print element_square
        
        try:
            scan['img_square'][element_square][0,0,0]
            #just see if it exists
        except KeyError:
            #print 'key error!'
            #print element_square
            #print scan['scaninfo']['imsize_square']
            scan['scaninfo']['recon_nimages']+=1
            scan['img_square'][element_square]=zeros(scan['scaninfo']['imsize_square'],complex)
            scan['dcomp_tmp'][element_square]=zeros(scan['scaninfo']['imsize_square'],complex)
            scan['scaninfo']['image_counter'][element_square]=1
            densitycomp[element_square]=zeros(scan['scaninfo']['imsize_square'])
            densctr[element_square]=0
        #print scan['img_square'][element_square][0,0,0]
        #print element_square
        #scan['img_square'][element]=zeros(scan['scaninfo']['imsize_square'],complex)
        #img_ptr=scan['img'][element]
        #print 'pre recon img square'
        #print [float(imsize[1])/float(imsize[0]),1]
        #print shape(scan['img'][element])
        #print scan['scaninfo']['imsize_square']
        scan['scaninfo']['image_counter'][element_square]+=1
        #print element_square
        if element_square[4]==0 and element_square[3]==1: #if it is b=0, but an average, we don't rotate it for some reason
            rotangle=0
            #print 'strange image'
        else:
            rotangle=180*element[2]/scan['scaninfo']['number_of_dynamic_scans']
        #rotangle=0
        #print scan['scaninfo']['number_of_dynamic_scans']
        if sumdynamics==1 or coilcombinealgorithm=='sum' or sumaverages==1:
            #print 'rotating'
            densctr[element_square]+=1
            for zidx in range(scan['scaninfo']['imsize_square'][2]):
                #add in image space
                #scan['img_square'][element_square][:,:,zidx]+=rotate(zoom(scan['img'][element][:,:,zidx].real,[float(imsize[1])/float(imsize[0]),1]),rotangle,(1,0),False)+1j*rotate(zoom(scan['img'][element][:,:,zidx].imag,[float(imsize[1])/float(imsize[0]),1]),rotangle,(1,0),False)
                #add in k-space, I SHOULD BE DOING GRIDDING HERE
                scan['img_square'][element_square][:,:,zidx]+=fftshift(ifftn(fftshift(rotate(zoom(scan['img'][element][:,:,zidx].real,[float(imsize[1])/float(imsize[0]),1]),rotangle,(1,0),False)+1j*rotate(zoom(scan['img'][element][:,:,zidx].imag,[float(imsize[1])/float(imsize[0]),1]),rotangle,(1,0),False))))
                #new blade combination method
                #tmp_square=zeros(scan['scaninfo']['imsize_square'][0:2],complex)
                #tmp=fftshift(ifftn(fftshift(scan['img'][element][:,:,zidx])))
                #tmp_square[offset[0]:offset[1],:]=tmp[scan['scaninfo']['ysize']/2+scan['scaninfo']['ky_range'][0]:scan['scaninfo']['ysize']/2+scan['scaninfo']['ky_range'][1],:]
                #scan['img_square'][element_square][:,:,zidx]+=rotate(tmp_square.real,rotangle,(1,0),False)+1j*rotate(tmp_square.imag,rotangle,(1,0),False)
                
                #print scan['scaninfo']['imsize'][0:2]
                #the first blade really shouln't be added!!!
                #if densitycomp[element_square][:,:,zidx].max()==1:
                densitycomp[element_square][:,:,zidx]+=abs(rotate(blade,rotangle,(1,0),False))
                scan['dcomp_tmp'][element_square][:,:,zidx]=densitycomp[element_square][:,:,zidx]
                
                #else:
                #    densitycomp[element_square][:,:,zidx]+=rotate(blade,rotangle,(1,0),False)
        #zoom(real(img_ptr),[4,1,1])+1j*zoom(imag(img_ptr),[4,1,1])
        #print shape(scan['img'][element])
    #scan['dcomp_tmp']=densitycomp
    for element in scan['img_square']: #to average we need to account summing
        #print element
        if densctr[element]==1:
            scan['img_square'][element]=fftshift(fftn(fftshift(scan['img_square'][element])))
            #print 'no density comp!'
        else:
            #scan['img_square'][element]=fftshift(fftn(fftshift(scan['img_square'][element])))
            scan['img_square'][element]=fftshift(fftn(fftshift(scan['img_square'][element]/(densitycomp[element]+1)))) #kspace addition
            #print 'density comp!'
        scan['densitycomp']=densitycomp[element_square]
        scan['img_square'][element]/=scan['scaninfo']['image_counter'][element]
        
        
      
    
        
    return scan
    
    
    
    
def philips_dic_recon_gridding(scan,coilcombinealgorithm='sum',sumdynamics=1,phasecorr=1,sumaverages=1,num_iters=10,minimize_distortion=True,dot_r_iter=0.05,stop_at_data=False,motion_correction=False):
    from numpy import zeros, interp, linspace, exp, pi, linspace, ones, conj, sort, array, nonzero, concatenate, meshgrid, shape, expand_dims, reshape, zeros_like, atleast_3d, tile, ones_like, isfinite, dot
    from numpy.fft import fftshift, ifftshift, ifft, ifftn, fftn
    from copy import copy
    from scipy.ndimage.interpolation import zoom, rotate
    from scipy.ndimage.filters import gaussian_filter1d
    from nfft_wrappers import init_nfft_2d, init_iterative_nfft_2d, grid_nfft_2d, iterative_nfft_2d, finalize_nfft_2d, finalize_iterative_nfft_2d, init_nnfft_2d, grid_nnfft_2d, finalize_nnfft_2d, init_iterative_nnfft_2d, iterative_nnfft_2d, grid_etp
    from nfft_helpers import propeller_sampling_location_generator, calc_density_weights, comp_fov
    from scipy.signal import medfilt2d, convolve
    from gc import collect
    from image_reg import mreg2d, mreg2d_mp
    from threading import Thread
    #from multiprocessing import Pool#, apply_async, close, join
    
    #scan['img']={}
    #scan['img_square']={}
    #scan['img_square_cg']={}
#    scan['data_tmp']={}
#    scan['data']={}
#    scan['phasecorr']={}
#    scan['frequencycorr']={}
    if phasecorr==1: #I doubt it would work without phase correction
        scan=phase_correct_philips(scan,phasecorr)
    #imsize=copy(scan['scaninfo']['ksize'])
    #imsize[1]=int(imsize[1]/scan['scaninfo']['kx_oversample_factor'])
    #scan['scaninfo']['imsize']=copy(imsize)
    imsize=scan['scaninfo']['imsize']
    blade=[imsize[1],imsize[0]] #because it is reversed and 3d in here
    if sumdynamics==1:
        nblades=copy(scan['scaninfo']['ndynamics'])
    else:
        nblades=1
    M_py=[blade[0],blade[1]*nblades]
    if imsize[1]>imsize[0]:
        print('Long Axis Propeller')
        scan['scaninfo']['is_SAP']=False
        ro_loc_pad=2
        ph_loc_pad=0
        damp_weight_perc=0.9
    else:
        print('Short Axis Propeller')
        scan['scaninfo']['is_SAP']=True
        ro_loc_pad=0
        ph_loc_pad=2
        damp_weight_perc=0.6


    
    #scan['scaninfo']['nslices'] #locations
    #scan['scaninfo']['ncoils'] #channel
    #scan['scaninfo']['ndynamics'] #dynamics
    #scan['scaninfo']['naverages'] #averages
    #scan['scaninfo']['nextra1'] #extra1
    #scan['scaninfo']['nextra2'] #extra2
    
    #print('beginning gridding'
    
    #blade image FOV    
    fov_rect=tile(atleast_3d(comp_fov([imsize[0],imsize[1]],thresh=1)),[1,1,imsize[2]])
    
    for element in scan['img']: #changed from kspace
        #print element
        element_square=copy(element)
        #if sumdynamics==1:
        #    element_square=(element_square[0],element_square[1],0,element_square[3],element_square[4],element_square[5])
        if coilcombinealgorithm=='sum':
            element_square=(0,element_square[1],element_square[2],element_square[3],element_square[4],element_square[5])
            
        if sumaverages==1:
            element_square=(element_square[0],element_square[1],element_square[2],0,element_square[4],element_square[5])
        #print element_square
        
        try:
            scan['data_tmp'][element_square]+=fftshift(fftn(fftshift(scan['img'][element]*fov_rect)))
        except:
            scan['data_tmp'][element_square]=fftshift(fftn(fftshift(scan['img'][element]*fov_rect)))
        #I need to deal with coil, dynamics, averages
        #[channel,location,dynamics,average,extra1,extra2]
    #for elem in scan['data_tmp']:
    #    print elem
        
    if sumdynamics==1:
        for element in scan['data_tmp']:
            if element[2]==0:
                for k in range(scan['scaninfo']['ndynamics']): #do dynamics
                    #element_square=copy(element)
                    element_square=(element[0],element[1],k,element[3],element[4],element[5])
                    try:
                        #print('concatenating',element_square)
                        scan['data'][element]=concatenate((scan['data'][element],scan['data_tmp'][element_square]),0)
                        #print shape(scan['data'][element])
                    except:
                        #print('initializing')
                        scan['data'][element]=scan['data_tmp'][element]
    else:
        scan['data']=scan['data_tmp']
    #for elem in scan['data']:
    #    print elem
    if stop_at_data==True:
        return scan
    
    
    #FOV support
    fov=comp_fov(scan['scaninfo']['imsize_square'][0:2],thresh=1)
    
    #x,y=meshgrid(linspace(-1,1,scan['scaninfo']['imsize_square'][0],False),linspace(-1,1,scan['scaninfo']['imsize_square'][1],False))
    #dist=pow(pow(x,2)+pow(y,2),0.5)
    #fov=zeros([scan['scaninfo']['imsize_square'][0],scan['scaninfo']['imsize_square'][1]])
    #fov[dist<=1]=1

    #image sampling locations
    
    
    #img_locs_init=copy(img_locs)
    del scan['data_tmp']
    del scan['datainfo']['auxiliary_phase_data']
    del scan['datainfo']['STD']
    del scan['datainfo']['NOI']
    del scan['datainfo']['REJ']
    del scan['datainfo']['PHX']
    del scan['datainfo']['NAV']
    del scan['datainfo']['FRX']
    del scan['phasecorr']
    del scan['frequencycorr']
    del scan['frequencycorr_orig']
    del scan['phasecorr_orig']
    #del scan['datainfo']
    del scan['dcomp_tmp']
    del scan['img']
    del scan['kspace']
    collect() #garbage collection!
    
    if scan['scaninfo']['fieldmap_filename']!='':
        x,y=meshgrid(linspace(-1,1,scan['scaninfo']['imsize_square'][0],False),linspace(-1,1,scan['scaninfo']['imsize_square'][1],False))
        img_locs=concatenate((expand_dims(y/2,3),expand_dims(x/2,3),expand_dims(x*0,3)),2)
        scan['fieldmap']=zoom(scan['fieldmap'],scan['scaninfo']['imsize_square'][0]/float(shape(scan['fieldmap'])[0])) #assume square...
        #scan['fieldmap']=convolve(scan['fieldmap'],ones([3,3]),'same')
        scan['fieldmap']=medfilt2d(scan['fieldmap'],7)
        #print shape(scan['fieldmap'])
        #print shape(img_locs)
        img_locs[:,:,2]=scan['fieldmap']/float(2*max(abs(scan['fieldmap'].flatten())))
        #print(min(img_locs[:,:,0].flatten()),max(img_locs[:,:,0].flatten()),min(img_locs[:,:,1].flatten()),max(img_locs[:,:,1].flatten()),min(img_locs[:,:,2].flatten()),max(img_locs[:,:,2].flatten()))
    

    #print blade
    #print nblades
    #loc_blade_old=-9999
    #allocated=False
    #p=Pool(2)
    t=[]
    loc_blade_old=[]
    for element in scan['data']:
        #print element
        #print(blade,nblades,element[2]/float(scan['scaninfo']['ndynamics']))
        try:
            scan['img_square'][element][0,0,0]
            #just see if it exists
        except KeyError:
            scan['img_square'][element]=zeros(scan['scaninfo']['imsize_square'],complex)
            scan['scaninfo']['recon_nimages']+=1
            scan['img_square_cg'][element]=zeros(scan['scaninfo']['imsize_square'],complex)
            
        #print(element[2]/float(scan['scaninfo']['ndynamics']))
        #if sumdynamics==1 and allocated==False:
        loc_blade=element[2]/float(scan['scaninfo']['ndynamics'])
        #print loc_blade
        if loc_blade!=loc_blade_old:
            #print('sampling locs')
            locs_vec,locs=propeller_sampling_location_generator(blade,nblades,loc_blade,sense_factor=scan['scaninfo']['sense_factor'])
            #print('after sampling locs')
            #print shape(locs)
            w=calc_density_weights(M_py,ro_loc_pad=ro_loc_pad,ph_loc_pad=ph_loc_pad,blade=blade,nblades=nblades,sense_factor=scan['scaninfo']['sense_factor']) #0 for ro_loc_pad
            #print(shape(w))
            w=medfilt2d(w,7)
            #scan['w_temp']=w
            
            #data consistency
            x,y=meshgrid(linspace(-1,1,blade[0],False),linspace(-1,1,blade[1],False))
            y=max(abs(y.copy().flatten()))-abs(y.copy())
            #print(shape(x>0.8))
            #print(shape(y))
            #damp errors during ramps
            y[abs(x)>damp_weight_perc]*=(1-abs(x[abs(x)>damp_weight_perc]))/(1-damp_weight_perc)
            y[y==0]=min(y.flatten()) #just make sure we don't hit 0 anywhere
            
            w_consistency=pow(tile(y.transpose(),[1,nblades]),2)
            scan['w_consistency']=w_consistency
        loc_blade_old=copy(loc_blade)
        #2D gridding initialization
        #print('initializing gridder')
        #print(blade,M_py,imsize)
        #print(shape(scan['data'][element][:,:,0].transpose()))
        #print(shape(w))
        #print(scan['scaninfo']['imsize_square'][0:2])
        #if sumdynamics==1 and allocated==False:
        if motion_correction==True:
            gridder_bladewise=grid_etp(scan['scaninfo']['imsize_square'][0:2],M_py,fov_support=tile(atleast_3d(fov),[1,1,nblades]),w=atleast_3d(ones_like(w)),coils=ones([1,1,1]),blade_phase=ones([1,1,nblades]),recon_blades=True)
            #gridder_bladewise.recon_blades=True
            #gridder_bladewise.nblades=nblades
            #print nblades
            gridder_bladewise.initialize_gridding(locs)  #here!
            #print gridder_bladewise.bladesize
            #print shape(gridder_bladewise.w)
            #print gridder_bladewise.gridder
            
            #gridder_pre=grid_etp(scan['scaninfo']['imsize_square'][0:2],M_py,fov_support=fov,w=atleast_3d(w),coils=ones([1,1,1]),blade_phase=ones([1,1,nblades]),use_bladewise=True)
            gridder_pre=grid_etp(scan['scaninfo']['imsize_square'][0:2],M_py,fov_support=fov,w=atleast_3d(w)) #why the extra dim?
        else:
            gridder=grid_etp(scan['scaninfo']['imsize_square'][0:2],M_py,fov_support=fov,w=w)
            #allocated=True
        #print('gridder initialize')
        
        #print('gridding loop')
        for zidx in range(scan['scaninfo']['imsize_square'][2]):
            if motion_correction==True:
                print('Registering')
                #print shape(scan['data'][element][:,:,zidx].transpose())
                raw_data=atleast_3d(scan['data'][element][:,:,zidx].transpose())
                blade_imgs=gridder_bladewise.grid(atleast_3d(raw_data))
                gridder_bladewise.finalize()
                #del gridder_bladewise
                #print shape(blade_imgs)
                #matrix1=mreg2d_mp(abs(blade_imgs[:,:,0]),abs(blade_imgs)) #cut first registration step
                #scan['img_blades']=blade_imgs
                #for m in range(shape(matrix1)[2]):
                #    print matrix1[:,:,m]
                #print matrix[:,:,3]
            else:
                raw_data=scan['data'][element][:,:,zidx].transpose()
            if scan['scaninfo']['fieldmap_filename']=='':
                #new gridding code
                if motion_correction==True:
                    gridder_pre.initialize_gridding(locs)#,defmat=matrix1)
                else:
                    gridder.initialize_gridding(locs)  #here!
                #print('where init should be!')
                #old gridding code
                #p=init_nfft_2d(locs,scan['scaninfo']['imsize_square'][0],fov)
                #ip=init_iterative_nfft_2d(p,w)
            
            
            else: #3D gridding initialization
                #new gridding code
                locs_full=concatenate((expand_dims(locs.real,2),expand_dims(locs.imag,2),expand_dims(reshape(locs_vec[:,2],shape(locs)),3)),2)
                gridder.initialize_gridding(locs_full,img_locs=img_locs)
                #old gridding code
    #            locs_full=concatenate((expand_dims(locs.real,2),expand_dims(locs.imag,2),expand_dims(reshape(locs_vec[:,2],shape(locs)),3)),2)
    #            #try:
    #                #    np.fov #see if it exists
    #                #    reinit_nnfft_2d(np,img_locs,locs_full)
    #                #except:
    #            np=init_nnfft_2d(img_locs,locs_full,scan['scaninfo']['imsize_square'],fov,time_res=32)
    #            inp=init_iterative_nnfft_2d(np,w)
            #print shape(scan['img_square'][element][:,:,zidx])
            #print shape(scan['data'][element][:,:,zidx].transpose())
            #print shape(w)
            #print shape(scan['data'][element][:,:,zidx].transpose()*w)
            #print shape(grid_nfft_2d(p,scan['data'][element][:,:,zidx].transpose()*w))
            
            #new gridding code
            #print(shape(scan['data'][element]))
            #print(shape(w))
            if motion_correction==True: #2 stage registration
                static_img=gridder_pre.grid(raw_data)
                #print shape(static_img)
                matrix=mreg2d(abs(static_img),abs(blade_imgs))#,matrix1)
                #matrix=zeros_like(matrix1)
                #for m in range(shape(matrix1)[2]):
                #    matrix[:,:,m]=dot(matrix1[:,:,m],matrix2[:,:,m])
                gridder_pre.finalize()
                del gridder_pre
                try:
                    t.join()
                    gridder.finalize()
                    #del gridder
                except:
                    print('gridder joining failed')
                    #pass
                gridder=grid_etp(scan['scaninfo']['imsize_square'][0:2],M_py,fov_support=fov,w=atleast_3d(w),coils=ones([1,1,1]),blade_phase=ones([1,1,nblades]),use_bladewise=True)
                gridder.initialize_gridding(locs,defmat=matrix)
                #for m in range(shape(matrix2)[2]):
                #    print matrix2[:,:,m]
                #for m in range(shape(matrix2)[2]):
                #    print matrix[:,:,m]

            #scan['img_square'][element][:,:,zidx]=0
            #print shape(raw_data)
            scan['img_square'][element][:,:,zidx]=gridder.grid(raw_data)
            #scan['igrid']=gridder.igrid(scan['img_square'][element][:,:,zidx])
            #scan['img_square'][element][:,:,zidx][~isfinite(scan['img_square'][element][:,:,zidx])]=0 #to fix strange nan errors in cg_solve
            #print shape(scan['img_square'][element])
            if minimize_distortion==True:
                gridder.w=w_consistency
                if motion_correction==True:
                    gridder.w=atleast_3d(gridder.w)
            scan['w']=w_consistency
            if num_iters==0:
                scan['img_square_cg'][element][:,:,zidx]=0
            else:
                if motion_correction==True:
                    elem_prev=(element[0],element[1],element[2],element[3],element[4],element[5])
                    t=Thread(target=cg_thread,args=(gridder,scan['img_square_cg'][elem_prev],zidx,raw_data,scan['img_square'][element][:,:,zidx]),kwargs={'verbose':True,'maxiter':num_iters}) #,kwds=(verbose=True,maxiter=num_iters)
                    t.start()
                else:
                    scan['img_square_cg'][element][:,:,zidx],niter_cg,resid_cg=gridder.cg_solve(raw_data,scan['img_square'][element][:,:,zidx],verbose=True,maxiter=num_iters)
                #print t.join()
                #scan['img_square_cg'][element][:,:,zidx],niter_cg,resid_cg=gridder.cg_solve(raw_data,scan['img_square'][element][:,:,zidx],verbose=True,maxiter=num_iters)
                
            
            #old gridding code
            #basic 2D gridding
#            if scan['scaninfo']['fieldmap_filename']=='':
#                scan['img_square'][element][:,:,zidx]=grid_nfft_2d(p,scan['data'][element][:,:,zidx].transpose()*w)
#                #iterate to fix density weighting issues
#                scan['img_square'][element][:,:,zidx]=iterative_nfft_2d(p,ip,scan['data'][element][:,:,zidx].transpose(),scan['img_square'][element][:,:,zidx],num_iters,dot_r_iter,1)
#            
#            else: #gridding to fix field map issues
#                scan['img_square'][element][:,:,zidx]=grid_nnfft_2d(np,scan['data'][element][:,:,zidx].transpose()*w)
#                scan['img_square'][element][:,:,zidx]=iterative_nnfft_2d(np,inp,scan['data'][element][:,:,zidx].transpose(),scan['img_square'][element][:,:,zidx],num_iters,dot_r_iter,1)
#                #print('test')
        #p.close()
        #p.join()
        #new gridding code
        #print('finalizing gridder')
        #if sumdynamics==0:
        if motion_correction==False:
            gridder.finalize()
            #del gridder
    if motion_correction==True:
        t.join()
        gridder.finalize()  #umm, here!
        #del gridder
        #old gridding code
#        if scan['scaninfo']['fieldmap_filename']=='':
#            finalize_nfft_2d(p)
#            finalize_iterative_nfft_2d(ip)
#        else:
#            finalize_nnfft_2d(np)
#            finalize_iterative_nfft_2d(inp)
                
    
    
    
    #gridder.finalize()
    #    p=init_nfft_2d(locs,N_py[0],fov)
    #ip=init_iterative_nfft_2d(p,w)
    #
    ##get the kspace data
    #data=igrid_nfft_2d(p,phan)
    #
    ##reconstruct the image
    #img=grid_nfft_2d(p,data*w)
    #kspace=fftshift(fftn(fftshift(img)))
    #
    ##iteratively reconstruc the image
    #img_iter=iterative_nfft_2d(p,ip,data,img,num_iters,dot_r_iter,1)
    #kspace_iter=fftshift(fftn(fftshift(img_iter)))
    #
    #
    ##deallocate memory
    #finalize_nfft_2d(p)
    #finalize_iterative_nfft_2d(ip)

    return scan
    
def cg_thread(gridder,outptr,zidx,raw_data,x,verbose=True,maxiter=20):
    img,niter_cg,resid_cg=gridder.cg_solve(raw_data,x,verbose=verbose,maxiter=maxiter)
    outptr[:,:,zidx]=img
    