#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 08:58:46 2019

@author: wexu
"""

import mne
import os.path as op
import numpy as np
from config_GP_Learn import MEG_data_path,group_name,Ids
from mne.preprocessing import read_ica
import pandas as pd

for subject_id in Ids:
    
    subject = group_name+"%d" % subject_id
    print("processing subject: %s" % subject)
    
    tasks=['AVLearn','AVLearn']
    days=[100,200]   
    
    for task,day in zip(tasks,days):

        fname=op.join(MEG_data_path,subject,task+'_%d'%(day+subject_id)+'_tsss_mc.fif')
        
        raw_tsss_mc=mne.io.read_raw_fif(fname,preload=True)    
        if op.isfile(fname.replace("tsss_mc", "annot")):
            raw_tsss_mc.set_annotations(mne.read_annotations(fname.replace("tsss_mc", "annot")))
            print("Annotion loaded!")
        raw_tsss_mc.filter(l_freq=1, h_freq=40.0,fir_design='firwin',n_jobs=-1)  # band-pass filter data      
        
                
        ica=read_ica(op.join(MEG_data_path,subject,task+'_%d'%(day+subject_id)+'-ica.fif'))
        ICA_reject_threshold=np.load(fname.replace("tsss_mc.fif",'ICA_reject_threshold.npy'))
        if op.isfile(fname.replace("tsss_mc.fif",'ICA_excludes.npy')):
            ica.exclude.extend(np.load(fname.replace("tsss_mc.fif",'ICA_excludes.npy')).tolist())
        
        events=pd.read_csv(fname.replace('tsss_mc.fif','events.csv'))[["stim_onset", "button_press", "trigger_code"]].astype(int).values
        epochs=mne.Epochs(raw_tsss_mc, events=events, tmin=-0.2, tmax=1,decim=5,reject=dict(grad=ICA_reject_threshold[0], mag=ICA_reject_threshold[1])).drop_bad()
                  
        eog_idx,eog_scores=ica.find_bads_eog(raw_tsss_mc)
        ica.plot_scores(eog_scores, exclude=eog_idx, labels='eog')    
        ica.exclude=list(set(ica.exclude+eog_idx))
        # reject by ICA components timecourse
        print(ica.exclude)
        ica.plot_sources(raw_tsss_mc,block = True)   

        print(ica.exclude)        
        ica.plot_components(inst=epochs)
        
        print(ica.exclude)
        ica.plot_sources(raw_tsss_mc,block = True)   

        np.save(fname.replace("tsss_mc.fif",'ICA_excludes.npy'),ica.exclude) 


