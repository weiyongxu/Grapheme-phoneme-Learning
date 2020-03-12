#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 13:17:27 2019

@author: wexu
"""

import mne
import os.path as op
from config_GP_Learn import MEG_data_path,MRI_data_path,group_name,Ids

for subject_id in Ids:
    
    tasks=['AVLearn','AVLearn']
    days=[100,200]   
    
    for task,day in zip(tasks,days):
        
        subject = group_name+"%d" % subject_id
        print("processing subject: %s" % subject)        
        fname=op.join(MEG_data_path,subject,task+'_%d'%(day+subject_id)+'_tsss_mc.fif')
        
        src = mne.setup_source_space(task+'_'+str(day+subject_id), spacing='ico5',subjects_dir=MRI_data_path,add_dist=True,n_jobs=-1)
        
        mne.write_source_spaces(fname.replace('tsss_mc.fif','ico5-src.fif'), src,overwrite=True)
        
        #without MRI
        info = mne.io.read_info(fname)
        trans_fname = fname.replace('_tsss_mc.fif','-trans.fif')
        bem_sol_file=op.join(MRI_data_path,task+'_'+str(day+subject_id),'bem','%s-inner_skull-bem-sol.fif'%(task+'_'+str(day+subject_id)))            
        
        fwd = mne.make_forward_solution(info, trans=trans_fname, src=src, bem=bem_sol_file,meg=True, eeg=False)
                                            
        mne.write_forward_solution(fname.replace('_tsss_mc.fif','-meg-ico5-fwd.fif'), fwd, overwrite=True)