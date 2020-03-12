#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 10:47:16 2019

@author: wexu
"""

import mne
import os.path as op
from config_GP_Learn import MEG_data_path,group_name,Ids
from mne.preprocessing import read_ica
import numpy as np
import pandas as pd


for subject_id in Ids:
    
    subject = group_name+"%d" % subject_id
    print("processing subject: %s" % subject)
    
    tasks=['AVLearn','AVLearn']
    days=[100,200]   
    
    for task,day in zip(tasks,days):
    
        fname=op.join(MEG_data_path,subject,task+'_%d'%(day+subject_id)+'_tsss_mc.fif')     
        raw_tsss_mc=mne.io.read_raw_fif(fname,preload=True)    
        raw_tsss_mc.filter(l_freq=None, h_freq=40.0,fir_design='firwin')  # low-pass filter data
        
        ica = read_ica(fname.replace("_tsss_mc", "-ica"))
        ica.exclude = np.load(fname.replace("tsss_mc.fif",'ICA_excludes.npy')).tolist()
        raw_tsss_mc = ica.apply(raw_tsss_mc, exclude=ica.exclude)
    
        events=pd.read_csv(fname.replace('tsss_mc.fif','events.csv'))           
        artifacts=pd.read_csv(fname.replace('tsss_mc.fif','artifacts.csv'))            
        df=pd.merge(events,artifacts,on='stim_onset')
        df['trigger_code2']=df['trigger_code']+df['block']*1000    
        df=df[df['artifacts']==False]
         
        events = df[["stim_onset", "button_press", "trigger_code2"]].astype(int).values
        
        picks = mne.pick_types(raw_tsss_mc.info, meg=True, eeg=False, stim=False, eog=True,exclude='bads')
        epochs = mne.Epochs(raw_tsss_mc, events, event_id=None, tmin=-0.15, tmax=1, picks=picks, baseline=(None,0), preload=True,reject=None,proj=True,metadata=df)        

        epochs.save(fname.replace("_tsss_mc", "-epo"))
        