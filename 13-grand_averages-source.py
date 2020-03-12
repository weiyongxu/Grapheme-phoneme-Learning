#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 10:29:39 2019

@author: wexu
"""

import os.path as op
import mne
from mne.parallel import parallel_func
from mne.minimum_norm import (make_inverse_operator, apply_inverse)
from config_GP_Learn import MEG_data_path,MRI_data_path,group_name,Ids  


cond_day1_AV=list([])
cond_day2_AV=list([])

cond_day1_FB=list([])
cond_day2_FB=list([])

conditions_AV=['/A','/V','/AV'] #'/UB/A','/UB/V','/UB/AVX','/LB/A','/LB/V','/LB/AVC','/LB/AVI',
conditions_FB=['/YES','/NO','/UNKNOWN'] 

for cond in conditions_AV:            
    cond_day1_AV.append(cond)
    cond_day2_AV.append(cond)

for cond in conditions_FB:            
    cond_day1_FB.append(cond)
    cond_day2_FB.append(cond)

x
def run_inverse(subject_id):
    
    tasks=['AVLearn','AVLearn']
    days=[100,200]
    
    for task,day in zip(tasks,days):

        subject = group_name+"%d" % subject_id
        print("processing subject: %s" % subject)
        fname=op.join(MEG_data_path,subject,task+'_%d'%(day+subject_id)+'_tsss_mc.fif')         
        
        if day==100:            
            evokeds_AV = mne.read_evokeds(fname.replace("_tsss_mc", "_GA-ave"),cond_day1_AV)
#            evokeds_FB = mne.read_evokeds(fname.replace("_tsss_mc", "_GA-ave"),cond_day1_FB)

        elif day==200:
            evokeds_AV = mne.read_evokeds(fname.replace("_tsss_mc", "_GA-ave"),cond_day2_AV)
#            evokeds_FB = mne.read_evokeds(fname.replace("_tsss_mc", "_GA-ave"),cond_day2_FB)

        evokeds_AV=[evk for evk in evokeds_AV ]
#        evokeds_FB=[evk for evk in evokeds_FB ]
                
        cov_AV = mne.read_cov(fname.replace('_tsss_mc.fif','_AV-cov.fif'))
#        cov_FB = mne.read_cov(fname.replace('_tsss_mc.fif','_FB-cov.fif'))
    
        forward = mne.read_forward_solution(fname.replace('_tsss_mc.fif','-meg-ico5-fwd.fif'))
    
        inverse_operator_AV = make_inverse_operator(evokeds_AV[0].info, forward, cov_AV, loose=1, depth=0.8)

#        inverse_operator_FB = make_inverse_operator(evokeds_FB[0].info, forward, cov_FB, loose=1, depth=0.8)
                    
        snr = 3.0
        lambda2 = 1.0 / snr ** 2
        
        methods=['dSPM']
        pick_ori=None
        for method in methods:
            
            for evoked in evokeds_AV:
                stc = apply_inverse(evoked, inverse_operator_AV, lambda2, method=method, pick_ori=pick_ori)
                stc_fsaverage = mne.compute_source_morph(stc,subjects_dir=MRI_data_path).apply(stc)
                
                stc_fsaverage.save(fname.replace('tsss_mc.fif',evoked.comment.replace('/','_')+'-'+method))
                
#            for evoked in evokeds_FB:
#                stc = apply_inverse(evoked, inverse_operator_FB, lambda2, method=method, pick_ori=pick_ori)              
#                stc_fsaverage = mne.compute_source_morph(stc,subjects_dir=MRI_data_path).apply(stc)
#                stc_fsaverage.save(fname.replace('tsss_mc.fif',evoked.comment.replace('/','_')+'-'+method))
parallel, run_func, _ = parallel_func(run_inverse, n_jobs=10)
parallel(run_func(subject_id) for subject_id in Ids)

# %% Visualization


import os.path as op
import mne
from mne.parallel import parallel_func
from mne.minimum_norm import (make_inverse_operator, apply_inverse)
from config_GP_Learn import MEG_data_path,MRI_data_path,group_name,Ids  


cond_day1_AV=list([])
cond_day2_AV=list([])

cond_day1_FB=list([])
cond_day2_FB=list([])

conditions_AV=['/A','/V','/AV'] #'/UB/A','/UB/V','/UB/AVX','/LB/A','/LB/V','/LB/AVC','/LB/AVI',
conditions_FB=['/YES','/NO','/UNKNOWN'] 

for cond in conditions_AV:            
    cond_day1_AV.append(cond)
    cond_day2_AV.append(cond)

for cond in conditions_FB:            
    cond_day1_FB.append(cond)
    cond_day2_FB.append(cond)
    
    
import numpy as np
from config_GP_Learn import Ids
task='AVLearn'
method='dSPM'

conditions=['/A','/V','/AV']
timess=[[0.117,0.209,0.360,0.482],[0.107,0.180,0.293,0.520],[0.120,0.185,0.320,0.569]]
limss=[(2.5,4,9),(2.5,4,9),(6,9,15)]

views=['med','lat','caudal']
for condition,times, lims in zip(conditions,timess,limss):

    stcs_D1 = list()
    stcs_D2 = list()
    
    for subject_id in Ids:
        
        subject = group_name+"%d" % subject_id
        print("processing subject: %s" % subject)   
        fname_D1=op.join(MEG_data_path,subject,task+'_%d'%(100+subject_id)+'_tsss_mc.fif')             
        stc_D1=mne.read_source_estimate(fname_D1.replace('tsss_mc.fif',condition.replace('/','_'))+'-'+method)        
        stcs_D1.append(stc_D1)
    
        fname_D2=op.join(MEG_data_path,subject,task+'_%d'%(200+subject_id)+'_tsss_mc.fif')             
        stc_D2=mne.read_source_estimate(fname_D2.replace('tsss_mc.fif',condition.replace('/','_'))+'-'+method)        
        stcs_D2.append(stc_D2)
      
    
    stcs=stcs_D1+stcs_D2         
    data = np.average([s.data for s in stcs], axis=0)
    stc_average = mne.SourceEstimate(data, stcs[0].vertices, stcs[0].tmin, stcs[0].tstep)

    for time in times:
        for view in views:
            brain = stc_average.plot(views=view, hemi='both', subject='fsaverage',subjects_dir=MRI_data_path,
                                     size=300,time_label='',
                                     background='white',
                                     colorbar=False,
                                     smoothing_steps=2,
                                     clim=dict(kind='value', lims=lims),
                                     initial_time=time, time_unit='s')            
            
            brain.save_single_image('/nashome1/wexu/Results/MNE_Results/AVLearn/'+condition+str(time)+'_'+view+'.pdf')
            brain.close()
            



for condition,times, lims in zip(conditions,timess,limss):

    stcs_D1 = list()
    stcs_D2 = list()
    
    for subject_id in Ids:
        
        subject = group_name+"%d" % subject_id
        print("processing subject: %s" % subject)   
        fname_D1=op.join(MEG_data_path,subject,task+'_%d'%(100+subject_id)+'_tsss_mc.fif')             
        stc_D1=mne.read_source_estimate(fname_D1.replace('tsss_mc.fif',condition.replace('/','_'))+'-'+method)        
        stcs_D1.append(stc_D1)
    
        fname_D2=op.join(MEG_data_path,subject,task+'_%d'%(200+subject_id)+'_tsss_mc.fif')             
        stc_D2=mne.read_source_estimate(fname_D2.replace('tsss_mc.fif',condition.replace('/','_'))+'-'+method)        
        stcs_D2.append(stc_D2)
      
    
    stcs=stcs_D1+stcs_D2         
    data = np.average([s.data for s in stcs], axis=0)
    stc_average = mne.SourceEstimate(data, stcs[0].vertices, stcs[0].tmin, stcs[0].tstep)


    brain = stc_average.plot(views='lat', hemi='both', subject='fsaverage',subjects_dir=MRI_data_path,
                             size=600,time_label='',
                             #background='white',
                             #foreground='black',
                             colorbar=True,
                             smoothing_steps=2,
                             clim=dict(kind='value', lims=lims),
                             initial_time=times[0], time_unit='s')            
    
    brain.save_single_image('/nashome1/wexu/Results/MNE_Results/AVLearn/'+condition+str(time)+'_'+view+'_colorbar2.pdf')
    brain.close()