#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 09:49:09 2022

@author: alexander
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 16:45:55 2021

@author: alexa
"""


import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time

import sys
sys.path.insert(0, "E:\GitHub\DigitalTwinInjectionMolding")
sys.path.insert(0, '/home/alexander/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/')

from DIM.models.model_structures import GRU
from DIM.models.injection_molding import QualityModel
from DIM.optim.common import BestFitRate
from DIM.optim.param_optim import parallel_mode
from DIM.miscellaneous.PreProcessing import arrange_data_for_ident, eliminate_outliers, LoadDynamicData


charges = list(range(1,275))

mode='quality'
split = 'all'
del_outl=True
# split = 'part'

# path = 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/data/Stoergroessen/20220504/Versuchsplan/normalized/'
path = '/home/alexander/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/normalized/'
    
   
u_inj= ['Q_Vol_ist', 'V_Screw_ist','p_wkz_ist','T_wkz_ist','p_inj_ist']
u_press= ['Q_Vol_ist', 'V_Screw_ist','p_wkz_ist','T_wkz_ist','p_inj_ist']
u_cool= ['Q_Vol_ist', 'V_Screw_ist','p_wkz_ist','T_wkz_ist','p_inj_ist']

u_lab = [u_inj,u_press,u_cool]
y_lab = ['Durchmesser_innen','Gewicht']


data_train,data_val = \
LoadDynamicData(path,charges,split,y_lab,u_lab,mode,del_outl)

# c0_train = [np.zeros((dim_c,1)) for i in range(0,len(data_train['data']))]
# c0_val = [np.zeros((dim_c,1)) for i in range(0,len(data_val['data']))] 

# data_train['init_state'] = c0_train
# data_val['init_state'] = c0_val

data = {'data_train':data_train,'data_val':data_val}

pkl.dump(data,open('Versuchsplan_original_normalized.data','wb'))
