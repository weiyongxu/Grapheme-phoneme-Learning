#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 08:59:07 2019

@author: wexu
"""
import os.path as op
from config_GP_Learn import MEG_data_path,group_name,Ids
import numpy as np
import pandas as pd


Group_ACCuracy_LB=[]
Group_ACCuracy_UB=[]
Group_RT_LB=[]
Group_RT_UB=[]

for subject_id in Ids:
    task='AVLearn'
    day=100    
    subject = group_name+"%d" % subject_id
    print("processing subject: %s" % subject)

    fname=op.join(MEG_data_path,subject,task+'_%d'%(day+subject_id)+'_tsss_mc.fif') 
    
    df=pd.read_csv(fname.replace('tsss_mc.fif','events.csv'))
    
    ACCuracy_LB_D1=df[(df['task']==1) &(df['Learnability']==1)].dropna(subset=['Test_correct']).groupby('block')['Test_correct'].mean().values
    ACCuracy_UB_D1=df[(df['task']==1) &(df['Learnability']==0)].dropna(subset=['Test_correct']).groupby('block')['Test_correct'].mean().values

    RT_LB_D1=df[(df['task']==1) & (df['RT_Outlier']==False)&(df['Learnability']==1)].dropna(subset=['Reaction_time']).groupby('block')['Reaction_time'].mean().values
    RT_UB_D1=df[(df['task']==1) & (df['RT_Outlier']==False)&(df['Learnability']==0)].dropna(subset=['Reaction_time']).groupby('block')['Reaction_time'].mean().values


    task='AVLearn'
    day=200    
    subject = group_name+"%d" % subject_id
    print("processing subject: %s" % subject)

    fname=op.join(MEG_data_path,subject,task+'_%d'%(day+subject_id)+'_tsss_mc.fif') 
    
    df=pd.read_csv(fname.replace('tsss_mc.fif','events.csv'))
    
    ACCuracy_LB_D2=df[(df['task']==1) &(df['Learnability']==1)].dropna(subset=['Test_correct']).groupby('block')['Test_correct'].mean().values
    ACCuracy_UB_D2=df[(df['task']==1) &(df['Learnability']==0)].dropna(subset=['Test_correct']).groupby('block')['Test_correct'].mean().values

    RT_LB_D2=df[(df['task']==1) & (df['RT_Outlier']==False)&(df['Learnability']==1)].dropna(subset=['Reaction_time']).groupby('block')['Reaction_time'].mean().values
    RT_UB_D2=df[(df['task']==1) & (df['RT_Outlier']==False)&(df['Learnability']==0)].dropna(subset=['Reaction_time']).groupby('block')['Reaction_time'].mean().values
    
    ACCuracy_LB=np.concatenate([ACCuracy_LB_D1,ACCuracy_LB_D2])
    ACCuracy_UB=np.concatenate([ACCuracy_UB_D1,ACCuracy_UB_D2])
    
    RT_LB=np.concatenate([RT_LB_D1,RT_LB_D2])
    RT_UB=np.concatenate([RT_UB_D1,RT_UB_D2])
    
    Group_ACCuracy_LB.append(ACCuracy_LB)
    Group_ACCuracy_UB.append(ACCuracy_UB)

    Group_RT_LB.append(RT_LB)
    Group_RT_UB.append(RT_UB)
    
Group_ACCuracy_LB=np.array(Group_ACCuracy_LB)
Group_ACCuracy_UB=np.array(Group_ACCuracy_UB)

Group_RT_LB=np.array(Group_RT_LB)
Group_RT_UB=np.array(Group_RT_UB)



import matplotlib.pyplot as plt
#Accuracy
X1 = np.arange(1, 19)
Y1=np.mean(Group_ACCuracy_LB,axis=0)
Y1_error=np.std(Group_ACCuracy_LB,axis=0)

X2 = np.arange(1, 19)
Y2=np.mean(Group_ACCuracy_UB,axis=0)
Y2_error=np.std(Group_ACCuracy_UB,axis=0)

fig, ax = plt.subplots(figsize=(12, 6))

kwargs = dict(capsize=2,elinewidth=1.1, linewidth=0.6, ms=7)

ax.errorbar(X1, Y1, yerr=Y1_error, fmt='-o',  **kwargs, label='Learnable')
ax.errorbar(X2, Y2, yerr=Y2_error, fmt='-^',  **kwargs, label='Control')

ax.legend(loc='best', frameon=True)

ax.set_title('Accuracy in Day 1 and Day 2', fontsize=14)
ax.set_xlabel('Block Index', fontsize=16)
ax.set_ylabel('Accuracy', fontsize=16)
ax.set_xlim(0, 19)
from matplotlib.ticker import MaxNLocator
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_ylim(0, 1.6);


#RT

X1 = np.arange(1, 19)
Y1=np.mean(Group_RT_LB,axis=0)
Y1_error=np.std(Group_RT_LB,axis=0)

X2 = np.arange(1, 19)
Y2=np.mean(Group_RT_UB,axis=0)
Y2_error=np.std(Group_RT_UB,axis=0)

fig, ax = plt.subplots(figsize=(12, 6))

kwargs = dict(capsize=2,elinewidth=1.1, linewidth=0.6, ms=7)

ax.errorbar(X1, Y1, yerr=Y1_error, fmt='-o',  **kwargs, label='Learnable')
ax.errorbar(X2, Y2, yerr=Y2_error, fmt='-^',  **kwargs, label='Control')

ax.legend(loc='best', frameon=True)

ax.set_title('Reaction time in Day 1 and Day 2', fontsize=14)
ax.set_xlabel('Block Index', fontsize=16)
ax.set_ylabel('Reaction time (ms)', fontsize=16)
ax.set_xlim(0, 19)
from matplotlib.ticker import MaxNLocator
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#ax.set_ylim(0, 1.6);