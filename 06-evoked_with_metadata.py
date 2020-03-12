#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 12:57:37 2019

@author: wexu
"""

import mne
import os.path as op
import numpy as np
from config_GP_Learn import MEG_data_path,group_name,Ids
import matplotlib.pyplot as plt

conditions=dict()
conditions['/UB/AVX']='Learnability==0 and A_index>0 and V_index>0'
conditions['/LB/AVC']='Congruency==True and A_index>0 and V_index>0 and V_overall_learn_index!=-1 and A_overall_learn_index!=-1'
conditions['/LB/AVI']='Congruency==False and A_index>0 and V_index>0 and V_overall_learn_index!=-1 and A_overall_learn_index!=-1'
conditions['/YES']    ='trigger_code==510 and V_overall_learn_index!=-1 and A_overall_learn_index!=-1'
conditions['/NO']     ='trigger_code==520 and V_overall_learn_index!=-1 and A_overall_learn_index!=-1'
conditions['/UNKNOWN']='trigger_code==530'

for subject_id in Ids:
    
    subject = group_name+"%d" % subject_id
    print("processing subject: %s" % subject)
    tasks=['AVLearn','AVLearn']
    days=[100,200]   
    
    for task,day in zip(tasks,days):

        fname=op.join(MEG_data_path,subject,task+'_%d'%(day+subject_id)+'_tsss_mc.fif')         
        epo=mne.read_epochs(fname.replace("_tsss_mc", "-epo"))
    
        epochs = dict()    
        #by Index 
        query='AV_overall_learn_index >={0} and AV_overall_learn_index <={1} and task=={2}'
        
        index_splits= [[0,1],[2,8],[9,24],[0,8]] if day==100 else [[2,12]]
        
        for idx in index_splits:
            for task, task_id in zip(['Learning','Testing'],[0,1]):
                for cond,cond_query in conditions.items():                                                
                    subset=epo[query.format(idx[0],idx[1],task_id)+" and "+cond_query]
                    epochs["/".join(('IDX'+str(index_splits.index(idx)),task))+cond] = subset
                   
        epo_Nave_avg =dict()

        for idx in index_splits:
            for task, task_id in zip(['Learning','Testing'],[0,1]):
                for cond,cond_query in conditions.items():                                                
                    eop_ave_nave_AV=int((len(epochs["/".join(('IDX'+str(index_splits.index(idx)),task))+'/LB/AVC'])+
                                      len(epochs["/".join(('IDX'+str(index_splits.index(idx)),task))+'/LB/AVI'])+
                                      len(epochs["/".join(('IDX'+str(index_splits.index(idx)),task))+'/UB/AVX']))/3)

                    epo_Nave_avg["/".join(('IDX'+str(index_splits.index(idx)),task))+'/LB/AVC']=eop_ave_nave_AV
                    epo_Nave_avg["/".join(('IDX'+str(index_splits.index(idx)),task))+'/LB/AVI']=eop_ave_nave_AV
                    epo_Nave_avg["/".join(('IDX'+str(index_splits.index(idx)),task))+'/UB/AVX']=eop_ave_nave_AV
                    
                    eop_ave_nave_FB=int((len(epochs["/".join(('IDX'+str(index_splits.index(idx)),task))+'/YES'])+
                                      len(epochs["/".join(('IDX'+str(index_splits.index(idx)),task))+'/NO'])+
                                      len(epochs["/".join(('IDX'+str(index_splits.index(idx)),task))+'/UNKNOWN']))/3)

                    epo_Nave_avg["/".join(('IDX'+str(index_splits.index(idx)),task))+'/YES']=eop_ave_nave_FB
                    epo_Nave_avg["/".join(('IDX'+str(index_splits.index(idx)),task))+'/NO']=eop_ave_nave_FB
                    epo_Nave_avg["/".join(('IDX'+str(index_splits.index(idx)),task))+'/UNKNOWN']=eop_ave_nave_FB

        
        epo_N  =dict()
        evokeds=dict()
        for epo_name,epo in epochs.items():
            evokeds[epo_name]=epo.average()
            evokeds[epo_name].comment=epo_name
            epo_N[epo_name]=len(epo)    

        
        plt.bar(*zip(*epo_N.items()))
        plt.savefig(fname.replace("tsss_mc.fif",'epo_N.png'))
        plt.close()
        
        mne.write_evokeds(fname.replace("_tsss_mc", "-ave"), list(evokeds.values()))
        np.save(fname.replace("tsss_mc.fif",'epo_Nave_avg.npy'), epo_Nave_avg) 