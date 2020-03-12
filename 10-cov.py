#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 14:01:20 2019

@author: wexu
"""
import mne
import os.path as op
from config_GP_Learn import MEG_data_path,group_name,Ids

for subject_id in Ids:
    
    subject = group_name+"%d" % subject_id
    print("processing subject: %s" % subject)
    tasks=['AVLearn','AVLearn']
    days=[100,200]   
    
    for task,day in zip(tasks,days):

        fname=op.join(MEG_data_path,subject,task+'_%d'%(day+subject_id)+'_tsss_mc.fif')         
        epo=mne.read_epochs(fname.replace("_tsss_mc", "-epo"))        
        epo.apply_baseline((None,0)) 

        subset_AV=epo['A_index>=0 and V_index>=0 and Learnability>=0'] #AV trials
        
        subset_FB=epo['trigger_code==510 or trigger_code==520 or trigger_code==530'] #AV trials
        
        # take care of noise cov
        cov_AV = mne.compute_covariance(subset_AV, tmin=None,tmax=0, method='shrunk',rank=None)
        cov_FB = mne.compute_covariance(subset_FB, tmin=None,tmax=0, method='shrunk',rank=None)

        cov_AV.save(fname.replace('_tsss_mc.fif','_AV-cov.fif'))
        cov_FB.save(fname.replace('_tsss_mc.fif','_FB-cov.fif'))
        