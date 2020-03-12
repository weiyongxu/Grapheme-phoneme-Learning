#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 13:29:12 2019

@author: wexu
"""

import os.path as op
import mne
import pandas as pd
from mne.parallel import parallel_func
from mne.minimum_norm import (make_inverse_operator, apply_inverse)
from config_GP_Learn import MEG_data_path,MRI_data_path,group_name,Ids
import numpy as np
import math
  


cond_day1_AV=list([])
cond_day2_AV=list([])

cond_day1_FB=list([])
cond_day2_FB=list([])

cond_day1_IT=list([])
cond_day2_IT=list([])

conditions_AV=['/UB/AVX','/LB/AVC','/LB/AVI'] 
conditions_FB=['/YES','/NO','/UNKNOWN'] 
conditions_IT=['RG_LB_Auditory','RG_LB_Visual','RG_LB_Interaction','RG_UB_Auditory','RG_UB_Visual','RG_UB_Interaction'] 

tasks=['Learning','Testing']
index_splits_day1= [[0,1],[2,8],[9,24],[0,8]]
index_splits_day2= [[2,12]]

index_splits_day1_RG= [[0,1],[2,8],[9,24],[0,8],[0,24]]  
index_splits_day2_RG= [[2,12]]

for idx in index_splits_day1:
    for task in tasks: 
        for cond in conditions_AV:            
            cond_day1_AV.append("/".join(('IDX'+str(index_splits_day1.index(idx)),task))+cond)

for idx in index_splits_day2:
    for task in tasks:                
        for cond in conditions_AV:
            cond_day2_AV.append("/".join(('IDX'+str(index_splits_day2.index(idx)),task))+cond)

for idx in index_splits_day1:
    for task in ['Learning']: 
        for cond in conditions_FB:            
            cond_day1_FB.append("/".join(('IDX'+str(index_splits_day1.index(idx)),task))+cond)

for idx in index_splits_day2:
    for task in ['Learning']:                
        for cond in conditions_FB:
            cond_day2_FB.append("/".join(('IDX'+str(index_splits_day2.index(idx)),task))+cond)

#Interaction effect
for idx in index_splits_day1_RG:
    for cond in conditions_IT:        
        cond_day1_IT.append("_".join((cond,('IDX'+str(index_splits_day1_RG.index(idx))))))

for idx in index_splits_day2_RG:
    for cond in conditions_IT:        
        cond_day2_IT.append("_".join((cond,('IDX'+str(index_splits_day2_RG.index(idx))))))

def run_inverse(subject_id):
    
    tasks=['AVLearn','AVLearn']
    days=[100,200]
    
    for task,day in zip(tasks,days):

        subject = group_name+"%d" % subject_id
        print("processing subject: %s" % subject)
        fname=op.join(MEG_data_path,subject,task+'_%d'%(day+subject_id)+'_tsss_mc.fif')         
        
        if day==100:            
            evokeds_AV = mne.read_evokeds(fname.replace("_tsss_mc", "-ave"),cond_day1_AV)
            evokeds_FB = mne.read_evokeds(fname.replace("_tsss_mc", "-ave"),cond_day1_FB)
            evokeds_IT = mne.read_evokeds(fname.replace('tsss_mc','evoked_RG'))

        elif day==200:
            evokeds_AV = mne.read_evokeds(fname.replace("_tsss_mc", "-ave"),cond_day2_AV)
            evokeds_FB = mne.read_evokeds(fname.replace("_tsss_mc", "-ave"),cond_day2_FB)
            evokeds_IT = mne.read_evokeds(fname.replace('tsss_mc','evoked_RG'))

        epo_Nave_avg=np.load(fname.replace("tsss_mc.fif",'epo_Nave_avg.npy'),allow_pickle=True).item()
        epo_Nave_avg_RG=np.load(fname.replace("tsss_mc.fif",'epo_Nave_avg_RG.npy'),allow_pickle=True).item()
            
        # only for nave>15
        evokeds_AV=[evk for evk in evokeds_AV if (epo_Nave_avg[evk.comment]>15)]
        evokeds_FB=[evk for evk in evokeds_FB if (epo_Nave_avg[evk.comment]>15)]
                
        cov_AV = mne.read_cov(fname.replace('_tsss_mc.fif','_AV-cov.fif'))
        cov_FB = mne.read_cov(fname.replace('_tsss_mc.fif','_FB-cov.fif'))
    
        forward = mne.read_forward_solution(fname.replace('_tsss_mc.fif','-meg-ico5-fwd.fif'))
    
        inverse_operator_AV = make_inverse_operator(evokeds_AV[0].info, forward, cov_AV, loose=1, depth=0.8)

        inverse_operator_FB = make_inverse_operator(evokeds_FB[0].info, forward, cov_FB, loose=1, depth=0.8)
        
        inverse_operator_IT = make_inverse_operator(evokeds_IT[0].info, forward, cov_AV, loose=1, depth=0.8)        

        labels = mne.read_labels_from_annot(subject=task+'_'+str(day+subject_id),parc ='aparc', subjects_dir=MRI_data_path)            
        label_names=[labels[indx].name for indx in range(len(labels))]

        STC_L=labels[0]+labels[60] #'bankssts-lh + superiortemporal-lh'
        STC_R=labels[1]+labels[61] #'bankssts-rh + superiortemporal-rh'

        labels.append(STC_L) 
        labels.append(STC_R)
        label_names.append(STC_L.name)
        label_names.append(STC_R.name)
               
        snr = 3.0
        lambda2 = 1.0 / snr ** 2
        
        methods=['dSPM','MNE']
        pick_ori=None
        mode='mean'
        for method in methods:
            
            for evoked in evokeds_AV:
                stc = apply_inverse(evoked.apply_baseline(baseline=(None,0)), inverse_operator_AV, lambda2, method=method, pick_ori=pick_ori)
                label_ts = mne.extract_label_time_course(stc, labels, inverse_operator_AV['src'],allow_empty=True,mode=mode)
                if method=='dSPM':
                    label_ts=label_ts*math.sqrt(epo_Nave_avg[evoked.comment])/math.sqrt(evoked.nave)                    
                pd.DataFrame(label_ts.T,index=stc.times, columns=label_names).to_csv(fname.replace('tsss_mc.fif',evoked.comment.replace('/','_')+'-'+method+'.csv'))
    
                stc_fsaverage = mne.compute_source_morph(stc,subjects_dir=MRI_data_path).apply(stc)
                if method=='dSPM':
                    stc_fsaverage=stc_fsaverage*math.sqrt(epo_Nave_avg[evoked.comment])/math.sqrt(evoked.nave)                                    
                stc_fsaverage.save(fname.replace('tsss_mc.fif',evoked.comment.replace('/','_')+'-'+method))
                
            for evoked in evokeds_FB:
                stc = apply_inverse(evoked.apply_baseline(baseline=(None,0)), inverse_operator_FB, lambda2, method=method, pick_ori=pick_ori)           
                label_ts = mne.extract_label_time_course(stc, labels, inverse_operator_FB['src'],allow_empty=True,mode=mode)             
                if method=='dSPM':
                    label_ts=label_ts*math.sqrt(epo_Nave_avg[evoked.comment])/math.sqrt(evoked.nave)                    
                pd.DataFrame(label_ts.T,index=stc.times, columns=label_names).to_csv(fname.replace('tsss_mc.fif',evoked.comment.replace('/','_')+'-'+method+'.csv'))
                           
                stc_fsaverage = mne.compute_source_morph(stc,subjects_dir=MRI_data_path).apply(stc)
                if method=='dSPM':
                    stc_fsaverage=stc_fsaverage*math.sqrt(epo_Nave_avg[evoked.comment])/math.sqrt(evoked.nave)                                                    
                stc_fsaverage.save(fname.replace('tsss_mc.fif',evoked.comment.replace('/','_')+'-'+method))

            for evoked in evokeds_IT:
                stc = apply_inverse(evoked.apply_baseline(baseline=(None,0)),inverse_operator_IT, lambda2, method=method, pick_ori=pick_ori)           
                label_ts = mne.extract_label_time_course(stc, labels, inverse_operator_IT['src'],allow_empty=True,mode=mode)             
                if method=='dSPM':
                    label_ts=label_ts*math.sqrt(epo_Nave_avg_RG[evoked.comment])/math.sqrt(evoked.nave)                    
                pd.DataFrame(label_ts.T,index=stc.times, columns=label_names).to_csv(fname.replace('tsss_mc.fif',evoked.comment.replace('/','_')+'-'+method+'.csv'))
                           
                stc_fsaverage = mne.compute_source_morph(stc,subjects_dir=MRI_data_path).apply(stc)
                if method=='dSPM':
                    stc_fsaverage=stc_fsaverage*math.sqrt(epo_Nave_avg_RG[evoked.comment])/math.sqrt(evoked.nave)                                                    
                stc_fsaverage.save(fname.replace('tsss_mc.fif',evoked.comment.replace('/','_')+'-'+method))


parallel, run_func, _ = parallel_func(run_inverse, n_jobs=10)
parallel(run_func(subject_id) for subject_id in Ids)
