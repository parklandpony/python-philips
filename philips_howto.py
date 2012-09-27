# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 13:30:56 2012

Philips raw data file read howto.

@author: eric
"""

from read_philips_list import read_philips_list, philips_dic_compactify, philips_dic_recon_basic, philips_dic_compactify_image, philips_dic_recon_gridding, phase_correct_philips

readtype='new'
save_image=1 #1 is niftii, 2 is dicom (non-functional because the library is stupid)
output_report=0
sumdynamics=1
phasecorr=1
sumaverages=0

#read raw data
params=read_philips_list('data_file_name.list','phase_correction_file.list') #raw data file first, phase correction file second (if needed)
#reconstruct propeller data (just an example because it won't work for you!)
params=philips_dic_recon_gridding(params,'sum',sumdynamics,phasecorr,sumaverages,num_iters=20,minimize_distortion=False,stop_at_data=False)
#get the data from the dictionary into a numpy matrix
img_square=philips_dic_compactify_image(params,'img_square')
img_square_cg=philips_dic_compactify_image(params,'img_square_cg')
data=philips_dic_compactify(params,'data').transpose([1,0,2,3])
    