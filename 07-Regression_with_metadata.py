#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 09:53:55 2019

@author: wexu
"""

import mne
import os.path as op
import numpy as np
from config_GP_Learn import MEG_data_path,group_name,Ids
import pandas as pd
from mne.stats.regression import linear_regression_raw
from mne.preprocessing import read_ica

LB_conditions=dict()
UB_conditions=dict()

UB_conditions['/UB/A']='Learnability==0 and A_index>0 and V_index==0'
UB_conditions['/UB/V']='Learnability==0 and A_index==0 and V_index>0'
UB_conditions['/UB/AVX']='Learnability==0 and A_index>0 and V_index>0'

LB_conditions['/LB/A']='Congruency==False and Learnability==1 and A_index>0 and V_index==0'
LB_conditions['/LB/V']='Congruency==False and Learnability==1 and A_index==0 and V_index>0'
LB_conditions['/LB/AVC']='Congruency==True and A_index>0 and V_index>0 and V_overall_learn_index!=-1 and A_overall_learn_index!=-1'

min_N=10

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
        ica.exclude = np.load(fname.replace("tsss_mc.fif",'ICA_excludes_EOG_EKG.npy')).tolist()     #EOG_EKG
        ica.exclude.extend(np.load(fname.replace("tsss_mc.fif",'ICA_excludes_alpha.npy')).tolist()) #alpha
        raw_tsss_mc = ica.apply(raw_tsss_mc, exclude=ica.exclude)
 
        
        events=pd.read_csv(fname.replace('tsss_mc.fif','events.csv'))           
        artifacts=pd.read_csv(fname.replace('tsss_mc.fif','artifacts.csv'))            
        df=pd.merge(events,artifacts,on='stim_onset')
        df['trigger_code2']=df['trigger_code']+df['block']*1000    
        df=df[df['artifacts']==False]

        epo_Nave_avg =dict()
        
        REG_list=list([])        
        #by Index 
        idx_query='AV_overall_learn_index >={0} and AV_overall_learn_index <={1} and task==0'
        
        index_splits= [[0,1],[2,8],[9,24],[0,8],[0,24]] if day==100 else [[2,12]]
        
        for idx in index_splits:
                               
            UB_A=df.query(UB_conditions['/UB/A']+" and "+idx_query.format(idx[0]/2,idx[1]/2))              
            UB_V=df.query(UB_conditions['/UB/V']+" and "+idx_query.format(idx[0]/2,idx[1]/2))
            UB_AVX=df.query(UB_conditions['/UB/AVX']+" and "+idx_query.format(idx[0],idx[1]))
            
            print(len(UB_A))
            print(len(UB_V))
            print(len(UB_AVX))


            LB_A=df.query(LB_conditions['/LB/A']+" and "+idx_query.format(idx[0]/2,idx[1]/2))              
            LB_V=df.query(LB_conditions['/LB/V']+" and "+idx_query.format(idx[0]/2,idx[1]/2))
            LB_AVC=df.query(LB_conditions['/LB/AVC']+" and "+idx_query.format(idx[0],idx[1]))

            print(len(LB_A))
            print(len(LB_V))
            print(len(LB_AVC))
            
            if len(LB_A)>min_N and len(LB_V)>min_N and len(LB_AVC)>min_N and len(UB_A)>min_N and len(UB_V)>min_N and len(UB_AVX)>min_N:
                
                RG_UB_events=pd.concat([UB_A,UB_V,UB_AVX])
                RG_UB_events.loc[:,'RG_UB_Auditory'+'_'+'IDX'+str(index_splits.index(idx))]=(RG_UB_events['A_index']>0).astype(int).values
                RG_UB_events.loc[:,'RG_UB_Visual'+'_'+'IDX'+str(index_splits.index(idx))]=(RG_UB_events['V_index']>0).astype(int).values
                RG_UB_events.loc[:,'RG_UB_Interaction'+'_'+'IDX'+str(index_splits.index(idx))]=((RG_UB_events['V_index']>0).values & (RG_UB_events['A_index']>0).values).astype(int)    
                RG_UB_events.loc[:,'Intercept']=1
                
                RG_UB_keys = [s+'_'+'IDX'+str(index_splits.index(idx)) for s in ['RG_UB_Auditory','RG_UB_Visual','RG_UB_Interaction']]
                RG_UB_events2 = RG_UB_events[["stim_onset", "button_press", "Intercept"]].astype(int).values
                RG_UB_covariates = RG_UB_events[RG_UB_keys]
                UB_RG=linear_regression_raw(raw_tsss_mc,events=RG_UB_events2,event_id={},covariates=RG_UB_covariates,tmin=-0.15,tmax=1.,reject=None)
                REG_list.extend([UB_RG[cond] for cond in RG_UB_keys])                
            
                RG_LB_events=pd.concat([LB_A,LB_V,LB_AVC])
                RG_LB_events.loc[:,'RG_LB_Auditory'+'_'+'IDX'+str(index_splits.index(idx))]=(RG_LB_events['A_index']>0).astype(int).values
                RG_LB_events.loc[:,'RG_LB_Visual'+'_'+'IDX'+str(index_splits.index(idx))]=(RG_LB_events['V_index']>0).astype(int).values
                RG_LB_events.loc[:,'RG_LB_Interaction'+'_'+'IDX'+str(index_splits.index(idx))]=((RG_LB_events['V_index']>0).values & (RG_LB_events['A_index']>0).values).astype(int)    
                RG_LB_events.loc[:,'Intercept']=1

                RG_LB_keys = [s+'_'+'IDX'+str(index_splits.index(idx)) for s in ['RG_LB_Auditory','RG_LB_Visual','RG_LB_Interaction']] 
                RG_LB_events2 = RG_LB_events[["stim_onset", "button_press", "Intercept"]].astype(int).values            
                RG_LB_covariates = RG_LB_events[RG_LB_keys]
                LB_RG=linear_regression_raw(raw_tsss_mc,events=RG_LB_events2,event_id={},covariates=RG_LB_covariates,tmin=-0.15,tmax=1.,reject=None)    
                REG_list.extend([LB_RG[cond] for cond in RG_LB_keys])

                eop_ave_nave_A=int((LB_RG['RG_LB_Auditory'+'_'+'IDX'+str(index_splits.index(idx))].nave+UB_RG['RG_UB_Auditory'+'_'+'IDX'+str(index_splits.index(idx))].nave)/2)
                epo_Nave_avg['RG_LB_Auditory'+'_'+'IDX'+str(index_splits.index(idx))]=eop_ave_nave_A
                epo_Nave_avg['RG_UB_Auditory'+'_'+'IDX'+str(index_splits.index(idx))]=eop_ave_nave_A
                
                eop_ave_nave_V=int((LB_RG['RG_LB_Visual'+'_'+'IDX'+str(index_splits.index(idx))].nave+UB_RG['RG_UB_Visual'+'_'+'IDX'+str(index_splits.index(idx))].nave)/2)
                epo_Nave_avg['RG_LB_Visual'+'_'+'IDX'+str(index_splits.index(idx))]=eop_ave_nave_V
                epo_Nave_avg['RG_UB_Visual'+'_'+'IDX'+str(index_splits.index(idx))]=eop_ave_nave_V

                eop_ave_nave_I=int((LB_RG['RG_LB_Interaction'+'_'+'IDX'+str(index_splits.index(idx))].nave+UB_RG['RG_UB_Interaction'+'_'+'IDX'+str(index_splits.index(idx))].nave)/2)
                epo_Nave_avg['RG_LB_Interaction'+'_'+'IDX'+str(index_splits.index(idx))]=eop_ave_nave_I
                epo_Nave_avg['RG_UB_Interaction'+'_'+'IDX'+str(index_splits.index(idx))]=eop_ave_nave_I
                
            else:
                epo_Nave_avg['RG_LB_Auditory'+'_'+'IDX'+str(index_splits.index(idx))]=0
                epo_Nave_avg['RG_UB_Auditory'+'_'+'IDX'+str(index_splits.index(idx))]=0                
                
                epo_Nave_avg['RG_LB_Visual'+'_'+'IDX'+str(index_splits.index(idx))]=0
                epo_Nave_avg['RG_UB_Visual'+'_'+'IDX'+str(index_splits.index(idx))]=0     
                
                epo_Nave_avg['RG_LB_Interaction'+'_'+'IDX'+str(index_splits.index(idx))]=0
                epo_Nave_avg['RG_UB_Interaction'+'_'+'IDX'+str(index_splits.index(idx))]=0     
                
        mne.write_evokeds(fname.replace('tsss_mc','evoked_RG'),REG_list)
        np.save(fname.replace("tsss_mc.fif",'epo_Nave_avg_RG.npy'), epo_Nave_avg)