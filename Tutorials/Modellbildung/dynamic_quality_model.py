#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 16:01:50 2022

@author: alexander
"""

from pathlib import Path
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np

path_dim = Path.cwd().parents[1]
sys.path.insert(0,path_dim.as_posix())


from DIM.models.models import GRU
from DIM.models.wrapper import QualityModel
from DIM.optim.param_optim import ParamOptimizer


dim_c = 1

# %% Load Data Manager
dm = pkl.load(open('/home/alexander/Desktop/DIM/data_manager.pkl','rb'))

# %% Get data for dynamic modelling
dm_data = dm.get_dynamic_modelling_data()

# %% Initial state is user-specified and must be appended manually
init_state = [np.zeros((dim_c,1)) for i in range(0,len(dm_data['data']))]
dm_data['init_state'] = init_state

# %% Divide randomly in training and validation data
idx_all = list(range(0,len(dm_data['data'])))
idx_train = list(np.random.choice(idx_all,size=int(len(idx_all)*0.7)))
idx_val = list(set(idx_all) - set(idx_train))

data_train = dm_data.copy()
data_train['data'] = [data_train['data'][i] for i in idx_train]
data_train['switch'] = [data_train['switch'][i] for i in idx_train]
data_train['init_state'] = [data_train['init_state'][i] for i in idx_train]

data_val = dm_data.copy()
data_val['data'] = [data_val['data'][i] for i in idx_val]
data_val['switch'] = [data_val['switch'][i] for i in idx_val]
data_val['init_state'] = [data_val['init_state'][i] for i in idx_val]


# %% Initialize model structure
u = ['p_wkz_ist','T_wkz_ist']
y = ['Gewicht']

model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=5,
                u_label=u,y_label=y,dim_out=1,name='q_model')

# %% Initialize optimizer

opts= {'initializations':10,'res_path': Path('home/alexander/Desktop/DIM/')}


opti = ParamOptimizer(model,data_train,data_val,**opts)

res = opti.optimize()
