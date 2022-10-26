#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 13:17:04 2020

@author: alexander
"""

import numpy as np
from scipy.io import loadmat  
import pandas as pd
import pickle as pkl

""" Load matlab data """

train_mat = loadmat('dataset3.mat')['dataset3']  # load mat-file
val_mat = loadmat('dataset2.mat')['dataset2']  # load mat-file
test_mat = loadmat('dataset1.mat')['dataset1']  # load mat-file


""" Convert to pandas dataframe"""

columns=['u','y']

train_pd = pd.DataFrame(data = train_mat, columns=columns)
val_pd = pd.DataFrame(data = val_mat, columns=columns)
test_pd = pd.DataFrame(data = test_mat, columns=columns)

""" Save """

pkl.dump(train_pd, open('training_data.pkl','wb'))
pkl.dump(val_pd, open('validation_data.pkl','wb'))
pkl.dump(test_pd, open('test_data.pkl','wb'))
