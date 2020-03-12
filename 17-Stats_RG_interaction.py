#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 19:36:56 2019

@author: wexu
"""

from config_GP_Learn import MEG_data_path,MRI_data_path,group_name,Ids
import os.path as op
import numpy as np
import mne
from mne.stats import summarize_clusters_stc

exclude = []  # Excluded subjects

day=100
task='AVLearn'

stat_tmin=0.5
stat_tmax=0.8
method='dSPM'
resamp_rate=200
tstep=1000/resamp_rate


cond_A='RG_LB_Interaction_IDX1'
cond_B='RG_UB_Interaction_IDX1'

stcs_A=list()
stcs_B=list()


for subject_id in Ids:
    
    if subject_id in exclude:
        continue
    
    subject = group_name+"%d" % subject_id
    print("processing subject: %s" % subject)
    fname=op.join(MEG_data_path,subject,task+'_%d'%(day+subject_id)+'_tsss_mc.fif')         
    
    if op.exists(fname.replace('tsss_mc.fif',cond_A.replace('/','_')+'-'+method+'-lh.stc')):
        
        morphed_A=mne.read_source_estimate(fname.replace('tsss_mc.fif',cond_A.replace('/','_')+'-'+method))
        morphed_B=mne.read_source_estimate(fname.replace('tsss_mc.fif',cond_B.replace('/','_')+'-'+method))
        
        morphed_A.crop(stat_tmin,stat_tmax).resample(resamp_rate)
        morphed_B.crop(stat_tmin,stat_tmax).resample(resamp_rate)
        
        stcs_A.append(morphed_A)
        stcs_B.append(morphed_B)
        



X = np.array([[c.data for c in stcs_A],[d.data for d in stcs_B]])
X = X[0,:, :, : ] - X[1,:, :, :]  # make paired contrast
X = np.transpose(X, [0, 2, 1])    # X needs to be samples (subjects) x time x space


from mne import (spatial_tris_connectivity, grade_to_tris)
from mne.stats import spatio_temporal_cluster_1samp_test,ttest_1samp_no_p
from scipy import stats as stats

tail=0
p_threshold=0.05

src_fname = '/nashome1/wexu/MNE_data/AVLearn/subjects/fsaverage/bem/fsaverage-ico-5-src.fif'
src = mne.read_source_spaces(src_fname)
connectivity = mne.spatial_src_connectivity(src)

#connectivity = spatial_tris_connectivity(grade_to_tris(5))
t_threshold = -stats.distributions.t.ppf(p_threshold / (1.+(tail==0)), len(X) - 1)
sigma = 1e-3  # sigma for the "hat" method
from functools import partial
stat_fun_hat = partial(ttest_1samp_no_p, sigma=sigma)
print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu = \
    spatio_temporal_cluster_1samp_test(X, connectivity=connectivity, n_jobs=20,stat_fun=stat_fun_hat,
                                  threshold=t_threshold,n_permutations=2000)
 

print('summary stats')
good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
for ind in good_cluster_inds:    
    inds_t, inds_v = clusters[ind]
    inds_t=inds_t*tstep
    inds_p=cluster_p_values[ind]
    print(' cluster   %d \n p value:  %f \n time:     %s \n clusters: %s '%(ind,inds_p,inds_t,inds_v))

x


print('Visualizing clusters.')
fsave_vertices = [np.arange(10242), np.arange(10242)]
stc_all_cluster_vis = summarize_clusters_stc(clu,tstep=0.005,tmin=stat_tmin, vertices=fsave_vertices,subject='fsaverage',p_thresh=0.05)

Folder_name='/nashome1/wexu/Results/MNE_Results/AVLearn/'
File_Name=cond_A.replace('_LB','')+'Day_'

filename= Folder_name+File_Name+str(day)+'.pickle'   
import pickle
with open(filename, 'wb') as f:
    pickle.dump(stc_all_cluster_vis, f)
    
for view in ['lat', 'med', 'ros', 'cau', 'dor', 'ven', 'fro', 'par']:

    brain = stc_all_cluster_vis.plot(hemi='lh', views=view,smoothing_steps=5,
                                     time_viewer=False,size=[800,800],
                                     subjects_dir=MRI_data_path,time_label='Duration significant (ms)',
                                     colormap='auto',background='white', foreground='black',
                                     )#clim=dict(kind='value',lims=[20, 40, 100])
    
    #indt=0
    #brain.set_data_time_index(indt)
    brain.save_single_image(Folder_name+File_Name+str(day)+'_'+view+'.pdf')
    brain.close()