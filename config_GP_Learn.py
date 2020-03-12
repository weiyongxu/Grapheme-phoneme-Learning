#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:49:21 2019

@author: wexu
"""

import os 

study_path='/nashome1/wexu/MNE_data/AVLearn'
group_name='AL'

MRI_data_path = os.path.join(study_path, 'subjects')
MEG_data_path = os.path.join(study_path, 'MEG')

os.environ["SUBJECTS_DIR"] = MRI_data_path

Ids=[1,3,4,5,6,7,8,11,12,13,14,16,17,18,19,20,21,22,23,24,25,27,28,29,31,32,33,34,35,36]#

delay=25 #ms