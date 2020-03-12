#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 10:36:12 2019

@author: wexu
"""
import mne
import os.path as op
import numpy as np
from config_GP_Learn import MEG_data_path,group_name,Ids
print(Ids)

tasks=['AVLearn','AVLearn']
days=[100,200]

for subject_id in Ids:
    
    subject = group_name+"%d" % subject_id
    print("processing subject: %s" % subject)
    
    for task,day in zip(tasks,days): 

        fname=op.join(MEG_data_path,subject,task+'_%d'%(day+subject_id)+'_tsss_mc.fif')
        RAW=mne.io.read_raw_fif(fname,preload=True)
        
        # noisy data segments related to movements should be manually annotated
        if op.isfile(fname.replace("tsss_mc", "annot")):
            RAW.set_annotations(mne.read_annotations(fname.replace("tsss_mc", "annot")))
            print("Annotion loaded!")
        
        RAW.filter(l_freq=1, h_freq=40.0,fir_design='firwin',n_jobs=-1)  # band-pass filter data      
        
        #cut the continuous MEG data info 1s epoch
        epochs=mne.Epochs(RAW, events=mne.make_fixed_length_events(RAW),tmin=0, tmax=1, baseline=None,reject=None,picks=mne.pick_types(RAW.info))
        
        ICA_reject_threshold = dict(grad=1500e-13, mag=4.00e-12)
        
        # after excluding the bad move artifacts, less than 5% of the data are defined as bad to select an threshold for ICA
        while mne.Epochs(RAW, events=mne.make_fixed_length_events(RAW), tmin=0, tmax=1, baseline=None,reject=ICA_reject_threshold).drop_bad().drop_log_stats(ignore=('BAD_move','BAD_ACQ_SKIP')) >=5:        
            ICA_reject_threshold['mag']=ICA_reject_threshold['mag']+0.25e-12
            
        print(ICA_reject_threshold)
        # see which channels are noisy
        epochs=mne.Epochs(RAW, events=mne.make_fixed_length_events(RAW), tmin=0, tmax=1, baseline=None,reject=ICA_reject_threshold).drop_bad()
        mne.viz.plot_drop_log(epochs.drop_log,subject=subject,show=False).savefig(fname.replace("tsss_mc.fif",'drop_bads_ICA.png'))
        
        np.save(fname.replace("tsss_mc.fif",'ICA_reject_threshold.npy'),np.array(list(ICA_reject_threshold.values())))