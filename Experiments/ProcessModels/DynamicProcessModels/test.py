#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 10:17:29 2022

@author: alexander
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

import sys
# sys.path.insert(0, "E:\GitHub\DigitalTwinInjectionMolding")
sys.path.insert(0, 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, '/home/alexander/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, 'E:/GitHub/DigitalTwinInjectionMolding/')

from DIM.miscellaneous.PreProcessing import LoadDynamicData
from DIM.models.model_structures import MLP
from DIM.models.injection_molding import ProcessModel
from DIM.optim.param_optim import parallel_mode
from DIM.optim.param_optim import ModelTraining
from DIM.optim.common import BestFitRate

path = '/home/alexander/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
split = 'all'
mode = 'process'

u_inj= ['v_inj_soll']
u_press= ['p_inj_soll']
u_cool= []

u_lab = [u_inj,u_press,u_cool]
y_lab = ['Q_Vol_ist','V_Screw_ist','p_wkz_ist','T_wkz_ist','p_inj_ist']

data,cycles_train_label,cycles_val_label,charge_train_label,charge_val_label = \
LoadDynamicData(path,[4],split,y_lab,u_lab,mode)

inj_model = MLP(dim_u=1,dim_out=5,dim_hidden=3,name='inj')
press_model = MLP(dim_u=1,dim_out=5,dim_hidden=3,name='press')
cool_model = MLP(dim_u=0,dim_out=5,dim_hidden=3,name='cool')

process_model = ProcessModel(subsystems=[inj_model,press_model,cool_model],
                              name='p_model')   

s_opts = {"max_iter": 1000, 'hessian_approximation':'exact'}

res =  ModelTraining(process_model,data,initializations=3, BFR=False, 
                  p_opts=None, s_opts=s_opts,mode='parallel')

process_model.SetParameters(res.loc[0]['params_val'])

_,e_val,_,y_val = parallel_mode(process_model,data['u_val'],data['y_val'],
                          data['init_state_val'],data['switch_val'])

plt.figure()
plt.plot(data['y_val'][0])
plt.title('True')

plt.figure()
plt.plot(np.array(y_val[0]))
plt.title('Est')

#  BFR auf einem Validierungsdatensatz
print(BestFitRate(np.array(y_val[1]),np.array(data['y_val'][1])))



