#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 10:29:54 2019

@author: wexu
"""
import mne
import pandas as pd
import os.path as op
import numpy as np
from config_GP_Learn import MEG_data_path,group_name,Ids
print(Ids)
from mne.preprocessing import read_ica

subject_id=[11]

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
        
        raw_tsss_mc.filter(l_freq=None,h_freq=40.0,fir_design='firwin',n_jobs=-1)  # band-pass filter data      

        df=pd.read_csv(fname.replace('tsss_mc.fif','events.csv'))
        events = df[["stim_onset", "button_press", "trigger_code"]].astype(int).values
        epochs=mne.Epochs(raw_tsss_mc, events=events, tmin=-0.2, tmax=1,preload=True,baseline=None)

        ica=read_ica(op.join(MEG_data_path,subject,task+'_%d'%(day+subject_id)+'-ica.fif'))
        ica.exclude = np.load(fname.replace("tsss_mc.fif",'ICA_excludes.npy')).tolist()
        
        ica.apply(epochs,exclude=ica.exclude)
        
        picks=mne.pick_channels(epochs.ch_names,[epochs.ch_names[0]]+epochs.ch_names[1::9]+epochs.ch_names[5::9]+epochs.ch_names[6::9])   
                
        epochs.drop_bad(reject=dict(grad=1500e-13, mag=5e-12))
        #manually remove bad trials after ICA        
        epochs.plot(n_channels=103,n_epochs=40,block = True,picks=picks, scalings=dict(mag=8e-12, grad=20e-11, eog=400e-5))

        epochs.plot_drop_log(subject=subject).savefig(fname.replace("tsss_mc.fif",'drop_bads_Manual.png'))
        
        df['artifacts']=~df['stim_onset'].isin(epochs.events[:,0])
        
        df.to_csv(fname.replace('tsss_mc.fif','events_with_artifacts_marked.csv'),index=False)
        