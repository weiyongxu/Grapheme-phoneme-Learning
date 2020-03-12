#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:55:29 2019

@author: wexu
"""

import mne
import os.path as op
import numpy as np
from config_GP_Learn import MEG_data_path,group_name,Ids
import matplotlib.pyplot as plt

conditions=dict()
conditions['/UB/A']='Learnability==0 and A_index>0 and V_index==0'
conditions['/UB/V']='Learnability==0 and A_index==0 and V_index>0'
conditions['/UB/AVX']='Learnability==0 and A_index>0 and V_index>0'

conditions['/LB/A']='Learnability==1 and A_index>0 and V_index==0'
conditions['/LB/V']='Learnability==1 and A_index==0 and V_index>0'
conditions['/LB/AVC']='Congruency==True and A_index>0 and V_index>0'
conditions['/LB/AVI']='Congruency==False and A_index>0 and V_index>0'

conditions['/A']='A_index>0 and V_index==0'
conditions['/V']='A_index==0 and V_index>0'
conditions['/AV']='A_index>0 and V_index>0'


conditions['/YES']    ='trigger_code==510'
conditions['/NO']     ='trigger_code==520'
conditions['/UNKNOWN']='trigger_code==530'

x

for subject_id in Ids:
    
    subject = group_name+"%d" % subject_id
    print("processing subject: %s" % subject)
    tasks=['AVLearn','AVLearn']
    days=[100,200]   
    
    for task,day in zip(tasks,days):

        fname=op.join(MEG_data_path,subject,task+'_%d'%(day+subject_id)+'_tsss_mc.fif')         
        epo=mne.read_epochs(fname.replace("_tsss_mc", "-epo"))
    
        evoked = dict()    

        for cond,cond_query in conditions.items():                                                
            evoked[cond] = epo[cond_query].average()
            evoked[cond].comment=cond
        
        mne.write_evokeds(fname.replace("_tsss_mc", "_GA-ave"), list(evoked.values()))

# %%

x
contrasts_day1=dict()
contrasts_day2=dict()

for cond in conditions:        
    contrasts_day1[cond]=list([])
    contrasts_day2[cond]=list([])


for subject_id in Ids:
    
    subject = group_name+"%d" % subject_id
    print("processing subject: %s" % subject)  
    task='AVLearn'

    day=100    
    fname=op.join(MEG_data_path,subject,task+'_%d'%(day+subject_id)+'_tsss_mc.fif')     
    ave_D1=mne.read_evokeds(fname.replace("_tsss_mc", "_GA-ave"))


    day=200    
    fname=op.join(MEG_data_path,subject,task+'_%d'%(day+subject_id)+'_tsss_mc.fif')     
    ave_D2=mne.read_evokeds(fname.replace("_tsss_mc", "_GA-ave"))
    
    for evk in ave_D1:
        contrasts_day1[evk.comment].append(evk)

    for evk in ave_D2:
        contrasts_day2[evk.comment].append(evk)


GA_D1=dict((k,mne.combine_evoked(v,'equal')) for k, v in contrasts_day1.items() if v)
GA_D2=dict((k,mne.combine_evoked(v,'equal')) for k, v in contrasts_day2.items() if v)


mne.viz.plot_evoked_topo([GA_D1['/UB/A'],GA_D1['/LB/A']],merge_grads=True)
mne.viz.plot_evoked_topo([GA_D2['/UB/A'],GA_D2['/LB/A']],merge_grads=True)

mne.viz.plot_evoked_topo([GA_D1['/UB/V'],GA_D1['/LB/V']],merge_grads=True)
mne.viz.plot_evoked_topo([GA_D2['/UB/V'],GA_D2['/LB/V']],merge_grads=True)

mne.viz.plot_evoked_topo([GA_D1['/UB/AVX'],GA_D1['/LB/AVC'],GA_D1['/LB/AVI']],merge_grads=True)
mne.viz.plot_evoked_topo([GA_D2['/UB/AVX'],GA_D2['/LB/AVC'],GA_D2['/LB/AVI']],merge_grads=True)

mne.viz.plot_evoked_topo([GA_D1['/LB/AVI'],GA_D1['/LB/AVC']],merge_grads=True)
mne.viz.plot_evoked_topo([GA_D2['/LB/AVI'],GA_D2['/LB/AVC']],merge_grads=True)


mne.viz.plot_evoked_topo([GA_D1['/YES'],GA_D1['/NO'],GA_D1['/UNKNOWN']],merge_grads=True)
mne.viz.plot_evoked_topo([GA_D2['/YES'],GA_D2['/NO'],GA_D2['/UNKNOWN']],merge_grads=True)


GA_A=mne.combine_evoked([GA_D1['/A'],GA_D2['/A']],'nave')
GA_A.pick_types('mag').plot_joint(title='Auditory',ts_args=dict(gfp=True),times=[0.117,0.209,0.360,0.482]).savefig('GA_'+'A'+'.pdf')

GA_V=mne.combine_evoked([GA_D1['/V'],GA_D2['/V']],'nave')
GA_V.pick_types('mag').plot_joint(title='Visual',ts_args=dict(gfp=True),times=[0.107,0.180,0.293,0.520]).savefig('GA_'+'V'+'.pdf')

#
GA_AV=mne.combine_evoked([GA_D1['/AV'],GA_D2['/AV']],'nave')
GA_AV.pick_types('mag').plot_joint(title='Audiovisual',ts_args=dict(gfp=True,ylim = dict(mag=[-180, 180])),times=[0.120,0.185,0.320,0.569]).savefig('GA_'+'AV'+'.pdf')
