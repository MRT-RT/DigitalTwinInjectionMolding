#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 16:45:32 2022

@author: alexander
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import pandas as pd

# Füge Pfad der Toolbox zum Suchpfad hinzu
path_dim = Path.cwd().parents[2]
sys.path.insert(0, path_dim.as_posix())

from DIM.optim.control_optim import QualityMultiStageOptimization
from DIM.miscellaneous.PreProcessing import LoadDynamicData

# %% Load data for debugging
# path_sys = 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/'
# path_sys = 'C:/Users/LocalAdmin/Documents/GitHub/DigitalTwinInjectionMolding/'
path_sys = '/home/alexander/GitHub/DigitalTwinInjectionMolding/' 
# path_sys = 'E:/GitHub/DigitalTwinInjectionMolding/'

path = path_sys + '/data/Versuchsplan/normalized/'

charges = list(range(1,10))

mode='quality'
split = 'all'
# split = 'part'
del_outl = True

u_inj= ['p_wkz_ist','T_wkz_ist']
u_press= ['p_wkz_ist','T_wkz_ist']
u_cool= ['p_wkz_ist','T_wkz_ist']

u_lab = [u_inj,u_press,u_cool]
y_lab = ['Durchmesser_innen']

data_train,data_val = LoadDynamicData(path,charges,split,y_lab,u_lab,
                                      mode,del_outl)

# %% Pick one example for testing

t1 = data_train['switch'][0][0]
t2 = data_train['switch'][0][1]

k1 = data_train['data'][0].index.get_loc(t1)
k2 = data_train['data'][0].index.get_loc(t2)
k3 = data_train['data'][0].index.get_loc(data_train['data'][0].index[-1])

# %% 

model = pkl.load(open('dynamic_PIM_model_c1.mod','rb'))
model.switching_instances = [k1,k2,k3]

init_inputs = data_train['data'][0][['T_wkz_ist','p_wkz_ist']]

res = QualityMultiStageOptimization(model,np.array([[1.1]]),init_inputs)


plt.plot(res['U'])
plt.plot(init_inputs.values)


