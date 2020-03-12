#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 16:37:22 2019

@author: wexu
"""

import mne
import os.path as op
from config_GP_Learn import MEG_data_path,MRI_data_path,group_name,Ids

# mne.create_default_subject(subjects_dir=MRI_data_path)
for subject_id in Ids:
    
    subject = group_name+"%d" % subject_id
    print("processing subject: %s" % subject)
    
    task='AVLearn'
    day=100  
    

    subject = group_name+"%d" % subject_id
    print("processing subject: %s" % subject)
    fname=op.join(MEG_data_path,subject,task+'_%d'%(day+subject_id)+'_tsss_mc.fif')
     
    mne.gui.coregistration(subjects_dir=MRI_data_path,subject='fsaverage',inst=fname) #use fsaverage template
    
        
    #check coreg
    trans = op.join(fname.replace('_tsss_mc.fif','-trans.fif'))
    info = mne.io.read_info(fname)
    aln = mne.viz.plot_alignment(info, trans, subject=task+'_'+str(day+subject_id), subjects_dir=MRI_data_path,dig=True,meg='helmet',surfaces=['brain','head'])
