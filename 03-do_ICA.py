#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:55:53 2019

@author: wexu
"""
import mne
import os.path as op
import numpy as np
from config_GP_Learn import MEG_data_path,group_name,Ids
from mne.preprocessing import ICA
print(Ids)

for subject_id in Ids:
    
    subject = group_name+"%d" % subject_id
    print("processing subject: %s" % subject)
    
    tasks=['AVLearn','AVLearn']
    days=[100,200]   
    
    for task,day in zip(tasks,days):
        
        fname=op.join(MEG_data_path,subject,task+'_%d'%(day+subject_id)+'_tsss_mc.fif')
        
        raw_tsss_mc=mne.io.read_raw_fif(fname,preload=True)    

        # noisy data segments related to movements should be manually annotated and excluded from ICA   
        if op.isfile(fname.replace("tsss_mc", "annot")):
            raw_tsss_mc.set_annotations(mne.read_annotations(fname.replace("tsss_mc", "annot")))
            print("Annotion loaded!")
        
        raw_tsss_mc.filter(l_freq=1, h_freq=40.0,fir_design='firwin',n_jobs=-1)  # band-pass filter data      
        #  define ICA parameters
        ica = ICA(method='fastica',n_components=0.99,max_iter=1000)            
        picks = mne.pick_types(raw_tsss_mc.info, meg=True, eeg=False, eog=False,stim=False, exclude='bads')
        # use the customized threshold
        ICA_reject_threshold=np.load(fname.replace("tsss_mc.fif",'ICA_reject_threshold.npy'))
        
        print(ICA_reject_threshold)
        
        ica.fit(raw_tsss_mc, picks=picks, reject=dict(grad=ICA_reject_threshold[0], mag=ICA_reject_threshold[1]),decim=5,reject_by_annotation=True) 
        ica.save(op.join(MEG_data_path , subject,task+'_%d'%(day+subject_id)+'-ica.fif'))
        