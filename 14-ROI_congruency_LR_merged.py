#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 21:42:48 2019

@author: wexu
"""

import mne
import os.path as op
import numpy as np
import pandas as pd
from config_GP_Learn import MEG_data_path,group_name,Ids


cond_day1_AV=dict()
cond_day2_AV=dict()

cond_day1_FB=dict()
cond_day2_FB=dict()

conditions_AV=['/UB/AVX','/LB/AVC','/LB/AVI'] 
conditions_FB=['/YES','/NO','/UNKNOWN'] 

tasks=['Learning','Testing']
index_splits_day1= [[0,1],[2,8],[9,24]]  
index_splits_day2= [[2,12]]


for idx in index_splits_day1:
    for task in tasks: 
        for cond in conditions_AV:            
            cond_day1_AV["/".join(('IDX'+str(index_splits_day1.index(idx)),task))+cond]=dict()

for idx in index_splits_day2:
    for task in tasks:                
        for cond in conditions_AV:
            cond_day2_AV["/".join(('IDX'+str(index_splits_day2.index(idx)),task))+cond]=dict()

for idx in index_splits_day1:
    for task in ['Learning']: 
        for cond in conditions_FB:            
            cond_day1_FB["/".join(('IDX'+str(index_splits_day1.index(idx)),task))+cond]=dict()

for idx in index_splits_day2:
    for task in ['Learning']:                
        for cond in conditions_FB:
            cond_day2_FB["/".join(('IDX'+str(index_splits_day2.index(idx)),task))+cond]=dict()

tasks=['AVLearn','AVLearn']
days=[100,200]
method='dSPM'

for subject_id in Ids:

    for task,day in zip(tasks,days):
    
        subject = group_name+"%d" % subject_id
        print("processing subject: %s" % subject)
        fname=op.join(MEG_data_path,subject,task+'_%d'%(day+subject_id)+'_tsss_mc.fif')         

                
        if day==100:
            for cond in cond_day1_AV.keys(): 
                if op.isfile(fname.replace('tsss_mc.fif',cond.replace('/','_')+'-'+method+'.csv')):
                    df=pd.read_csv(fname.replace('tsss_mc.fif',cond.replace('/','_')+'-'+method+'.csv'),index_col=0)
                    cond_day1_AV[cond][subject]=df
                    
            for cond in cond_day1_FB.keys():            
                if op.isfile(fname.replace('tsss_mc.fif',cond.replace('/','_')+'-'+method+'.csv')):
                    df=pd.read_csv(fname.replace('tsss_mc.fif',cond.replace('/','_')+'-'+method+'.csv'),index_col=0)
                    cond_day1_FB[cond][subject]=df

        elif day==200:
            
            for cond in cond_day2_AV.keys(): 
                if op.isfile(fname.replace('tsss_mc.fif',cond.replace('/','_')+'-'+method+'.csv')):
                    df=pd.read_csv(fname.replace('tsss_mc.fif',cond.replace('/','_')+'-'+method+'.csv'),index_col=0)
                    cond_day2_AV[cond][subject]=df
                    
            for cond in cond_day2_FB.keys():            
                if op.isfile(fname.replace('tsss_mc.fif',cond.replace('/','_')+'-'+method+'.csv')):
                    df=pd.read_csv(fname.replace('tsss_mc.fif',cond.replace('/','_')+'-'+method+'.csv'),index_col=0)
                    cond_day2_FB[cond][subject]=df


for cond in cond_day1_AV.keys():             
    cond_day1_AV[cond]=pd.Panel(cond_day1_AV[cond])
for cond in cond_day2_AV.keys():             
    cond_day2_AV[cond]=pd.Panel(cond_day2_AV[cond])

for cond in cond_day1_FB.keys():             
    cond_day1_FB[cond]=pd.Panel(cond_day1_FB[cond])
for cond in cond_day2_FB.keys():             
    cond_day2_FB[cond]=pd.Panel(cond_day2_FB[cond])


#region='bankssts-rh + superiortemporal-rh'

#revision 1 chage to bankssts
region='bankssts-rh'   
   

cond1_mean=dict()
cond2_mean=dict()
cond3_mean=dict()

cond1_std=dict()
cond2_std=dict()
cond3_std=dict()

cond_export=dict()

for day, phase ,cond_list in zip(['/D1/','/D1/','/D1/','/D2/'],['0','1','2','0'],[cond_day1_AV,cond_day1_AV,cond_day1_AV,cond_day2_AV]):
    
    #change condition to either Learning or Testing
    cond1='IDX/Learning/LB/AVC'.replace('IDX','IDX'+phase)   
    cond2='IDX/Learning/LB/AVI'.replace('IDX','IDX'+phase) 
    cond3='IDX/Learning/UB/AVX'.replace('IDX','IDX'+phase) 
   
    cond1_mean[day+cond1]=cond_list[cond1][:,0.5:0.8,region].mean(axis=0).mean(axis=0)  
    cond2_mean[day+cond2]=cond_list[cond2][:,0.5:0.8,region].mean(axis=0).mean(axis=0)  
    cond3_mean[day+cond3]=cond_list[cond3][:,0.5:0.8,region].mean(axis=0).mean(axis=0)  

    cond1_std[day+cond1]=cond_list[cond1][:,0.5:0.8,region].mean(axis=0).std(axis=0)/np.sqrt(cond_list[cond1].shape[0])  
    cond2_std[day+cond2]=cond_list[cond2][:,0.5:0.8,region].mean(axis=0).std(axis=0)/np.sqrt(cond_list[cond2].shape[0])   
    cond3_std[day+cond3]=cond_list[cond3][:,0.5:0.8,region].mean(axis=0).std(axis=0)/np.sqrt(cond_list[cond3].shape[0]) 
    
    cond_export[day+cond1]=cond_list[cond1][:,0.5:0.8,region].mean(axis=0)
    cond_export[day+cond2]=cond_list[cond2][:,0.5:0.8,region].mean(axis=0)
    cond_export[day+cond3]=cond_list[cond3][:,0.5:0.8,region].mean(axis=0)


for cond_key, cond_values in cond_export.items():
    cond_values.to_csv((cond_key+'/'+region+'.csv').replace('/','_'))

df_mean=pd.DataFrame([cond1_mean.values(),cond2_mean.values(),cond3_mean.values()],index=['AVC','AVI','AVX'],columns=['D1/0','D1/1','D1/2','D2/0',]).transpose()
df_std =pd.DataFrame([cond1_std.values(),cond2_std.values(),cond3_std.values()],index=['AVC','AVI','AVX'],columns=['D1/0','D1/1','D1/2','D2/0',]).transpose()

df_mean.plot(kind='bar',legend=True,yerr=df_std,title=region)

