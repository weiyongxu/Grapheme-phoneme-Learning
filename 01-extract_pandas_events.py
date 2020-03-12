#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:28:54 2019

@author: wexu
"""
import os.path as op
from config_GP_Learn import MEG_data_path,group_name,Ids
from Custom_Function import event_to_pandas

print(Ids)

for subject_id in Ids:

    subject = group_name+"%d" % subject_id
    print("processing subject: %s" % subject)
    
    tasks=['AVLearn','AVLearn']
    days=[100,200]
    
    
    for task,day in zip(tasks,days):
        
        fname=op.join(MEG_data_path,subject,task+'_%d'%(day+subject_id)+'_tsss_mc.fif')
        
        df=event_to_pandas(fname) #define events and extract related information
        
        df.to_csv(fname.replace('tsss_mc.fif','events.csv'),index=False)        