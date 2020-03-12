#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 10:00:21 2019

@author: wexu
"""

import os.path as op
from config_GP_Learn import MEG_data_path,group_name,Ids
import numpy as np
import pandas as pd

Group_RT_LB_mean=[]
Group_RT_UB_mean=[]

Group_RT_LB_std=[]
Group_RT_UB_std=[]

Group_RT_LB_mean_D2=[]
Group_RT_UB_mean_D2=[]

Group_RT_LB_std_D2=[]
Group_RT_UB_std_D2=[]

for subject_id in Ids:
    task='AVLearn'
    day=100    
    subject = group_name+"%d" % subject_id
    print("processing subject: %s" % subject)

    fname=op.join(MEG_data_path,subject,task+'_%d'%(day+subject_id)+'_tsss_mc.fif') 
    
    df=pd.read_csv(fname.replace('tsss_mc.fif','events.csv'))
    
    df['AV_overall_learn_index']=(df['AV_overall_learn_index']/2).astype(int)
    
    RT_LB_D1_mean=df[(df['task']==1) & (df['RT_Outlier']==False) & (df['Learnability']==1) & (df['A_overall_learn_index']!=-1) & (df['V_overall_learn_index']!=-1)].dropna(subset=['Reaction_time']).groupby('AV_overall_learn_index')['Reaction_time'].mean()
    RT_UB_D1_mean=df[(df['task']==1) & (df['RT_Outlier']==False) & (df['Learnability']==0) & (df['A_overall_learn_index']!=-1) & (df['V_overall_learn_index']!=-1)].dropna(subset=['Reaction_time']).groupby('AV_overall_learn_index')['Reaction_time'].mean()

    RT_LB_D1_std=df[(df['task']==1) & (df['RT_Outlier']==False) & (df['Learnability']==1) & (df['A_overall_learn_index']!=-1) & (df['V_overall_learn_index']!=-1)].dropna(subset=['Reaction_time']).groupby('AV_overall_learn_index')['Reaction_time'].std()
    RT_UB_D1_std=df[(df['task']==1) & (df['RT_Outlier']==False) & (df['Learnability']==0) & (df['A_overall_learn_index']!=-1) & (df['V_overall_learn_index']!=-1)].dropna(subset=['Reaction_time']).groupby('AV_overall_learn_index')['Reaction_time'].std()


    task='AVLearn'
    day=200    
    subject = group_name+"%d" % subject_id
    print("processing subject: %s" % subject)

    fname=op.join(MEG_data_path,subject,task+'_%d'%(day+subject_id)+'_tsss_mc.fif') 
    
    df=pd.read_csv(fname.replace('tsss_mc.fif','events.csv'))
    
    df['AV_overall_learn_index']=(df['AV_overall_learn_index']/2).astype(int)
    
    RT_LB_D2_mean=df[(df['task']==1) & (df['RT_Outlier']==False)&(df['Learnability']==1) & (df['A_overall_learn_index']!=-1) & (df['V_overall_learn_index']!=-1)].dropna(subset=['Reaction_time']).groupby('AV_overall_learn_index')['Reaction_time'].mean()
    RT_UB_D2_mean=df[(df['task']==1) & (df['RT_Outlier']==False)&(df['Learnability']==0) & (df['A_overall_learn_index']!=-1) & (df['V_overall_learn_index']!=-1)].dropna(subset=['Reaction_time']).groupby('AV_overall_learn_index')['Reaction_time'].mean()

    RT_LB_D2_std=df[(df['task']==1) & (df['RT_Outlier']==False)&(df['Learnability']==1) & (df['A_overall_learn_index']!=-1) & (df['V_overall_learn_index']!=-1)].dropna(subset=['Reaction_time']).groupby('AV_overall_learn_index')['Reaction_time'].std()
    RT_UB_D2_std=df[(df['task']==1) & (df['RT_Outlier']==False)&(df['Learnability']==0) & (df['A_overall_learn_index']!=-1) & (df['V_overall_learn_index']!=-1)].dropna(subset=['Reaction_time']).groupby('AV_overall_learn_index')['Reaction_time'].std()
 
    
    Group_RT_LB_mean.append(RT_LB_D1_mean)
    Group_RT_UB_mean.append(RT_UB_D1_mean)
    
    Group_RT_LB_std.append(RT_LB_D1_std)
    Group_RT_UB_std.append(RT_UB_D1_std)

    Group_RT_LB_mean_D2.append(RT_LB_D2_mean)
    Group_RT_UB_mean_D2.append(RT_UB_D2_mean)
    
    Group_RT_LB_std_D2.append(RT_LB_D2_std)
    Group_RT_UB_std_D2.append(RT_UB_D2_std)

  
Group_RT_LB_mean=pd.concat(Group_RT_LB_mean,axis=1).mean(axis=1)
Group_RT_UB_mean=pd.concat(Group_RT_UB_mean,axis=1).mean(axis=1)

Group_RT_LB_std=pd.concat(Group_RT_LB_std,axis=1).mean(axis=1)
Group_RT_UB_std=pd.concat(Group_RT_UB_std,axis=1).mean(axis=1)

Group_RT_LB_mean_D2=pd.concat(Group_RT_LB_mean_D2,axis=1).mean(axis=1)
Group_RT_UB_mean_D2=pd.concat(Group_RT_UB_mean_D2,axis=1).mean(axis=1)

Group_RT_LB_std_D2=pd.concat(Group_RT_LB_std_D2,axis=1).mean(axis=1)
Group_RT_UB_std_D2=pd.concat(Group_RT_UB_std_D2,axis=1).mean(axis=1)


import matplotlib.pyplot as plt

#RT

X1 = np.arange(0, 13)
Y1=np.array(Group_RT_LB_mean)
Y1_error=np.array(Group_RT_LB_std)

X2 = np.arange(0, 13)
Y2=np.array(Group_RT_UB_mean)
Y2_error=np.array(Group_RT_UB_std)

fig, ax = plt.subplots(figsize=(10, 6))

kwargs = dict(capsize=2,elinewidth=1.1, linewidth=0.6, ms=7)

ax.errorbar(X1, Y1, yerr=Y1_error, fmt='-o',  **kwargs, label='Learnable')
ax.errorbar(X2, Y2, yerr=Y2_error, fmt='-^',  **kwargs, label='Non-Learnable')

ax.legend(loc='best', frameon=True)

ax.set_title('Reaction time in Day 1', fontsize=14)
ax.set_xlabel('Learning Index', fontsize=12)
ax.set_ylabel('RT', fontsize=12)
ax.set_xlim(-1, 14)
from matplotlib.ticker import MaxNLocator
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#ax.set_ylim(0, 1.6);



X1 = np.arange(1, 7)
Y1=np.array(Group_RT_LB_mean_D2)[1:]
Y1_error=np.array(Group_RT_LB_std_D2)[1:]

X2 = np.arange(1, 7)
Y2=np.array(Group_RT_UB_mean_D2)[1:]
Y2_error=np.array(Group_RT_UB_std_D2)[1:]

fig, ax = plt.subplots(figsize=(10, 6))

kwargs = dict(capsize=2,elinewidth=1.1, linewidth=0.6, ms=7)

ax.errorbar(X1, Y1, yerr=Y1_error, fmt='-o',  **kwargs, label='Learnable')
ax.errorbar(X2, Y2, yerr=Y2_error, fmt='-^',  **kwargs, label='Non-Learnable')

ax.legend(loc='best', frameon=True)

ax.set_title('Reaction time in Day 2', fontsize=14)
ax.set_xlabel('Learning Index', fontsize=12)
ax.set_ylabel('RT', fontsize=12)
ax.set_xlim(0, 7)
from matplotlib.ticker import MaxNLocator
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#ax.set_ylim(0, 1.6);




X1 = np.arange(0, 13)
Y1=np.array(Group_RT_LB_mean)
Y1_error=np.array(Group_RT_LB_std)

X2 = np.arange(0, 13)
Y2=np.array(Group_RT_UB_mean)
Y2_error=np.array(Group_RT_UB_std)

X3 = np.arange(13, 19)
Y3=np.array(Group_RT_LB_mean_D2)[1:]
Y3_error=np.array(Group_RT_LB_std_D2)[1:]

X4 = np.arange(13, 19)
Y4=np.array(Group_RT_UB_mean_D2)[1:]
Y4_error=np.array(Group_RT_UB_std_D2)[1:]

X5=np.concatenate((X1,X3))
Y5=np.concatenate((Y1,Y3))
Y5_error=np.concatenate((Y1_error,Y3_error))

X6=np.concatenate((X2,X4))
Y6=np.concatenate((Y2,Y4))
Y6_error=np.concatenate((Y2_error,Y4_error))


fig, ax = plt.subplots(figsize=(10, 6))

kwargs = dict(capsize=2,elinewidth=1.1, linewidth=0.6, ms=7)

ax.errorbar(X5, Y5, yerr=Y5_error, fmt='-o',  **kwargs, label='Learnable')
ax.errorbar(X6, Y6, yerr=Y6_error, fmt='-^',  **kwargs, label='Control')

ax.legend(loc='best', frameon=True)

ax.set_title('Reaction time in Day 1 and Day 2', fontsize=14)
ax.set_xlabel('Learning Index', fontsize=16)
ax.set_ylabel('Reaction time (ms)', fontsize=16)
ax.set_xlim(-1, 19)
ax.set_ylim(-500, 4500)
from matplotlib.ticker import MaxNLocator
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#ax.set_ylim(0, 1.6);