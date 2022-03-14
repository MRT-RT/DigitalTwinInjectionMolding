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

# x0_train = [cycle[0][0,:] for cycle in data['y_train']]
# x0_val = [cycle[0][0,:] for cycle in data['y_val']]

# data['init_state_train'] = x0_train
# data['init_state_val'] = x0_val

inj_model = MLP(dim_u=1,dim_out=5,dim_hidden=1,name='inj')
press_model = MLP(dim_u=1,dim_out=5,dim_hidden=1,name='press')
cool_model = MLP(dim_u=0,dim_out=5,dim_hidden=1,name='cool')

process_model = ProcessModel(subsystems=[inj_model,press_model,cool_model],
                              name='p_model')   

# _,e_train,_,y_train = parallel_mode(process_model,data['u_train'],data['y_train'],
#                           data['init_state_train'],data['switch_train']) 


s_opts = {"max_iter": 1000, 'hessian_approximation':'limited-memory'}

res =  ModelTraining(process_model,data,initializations=1, BFR=False, 
                  p_opts=None, s_opts=None,mode='parallel')


# c1 = pkl.load(open(path+'cycle1.pkl','rb'))


# c1_min = c1.min()
# c1_max = c1.max()


# c1 = (c1-c1_min)/(c1_max-c1_min)

# c1['v_inj_soll'] = 1

# plt.plot(c1['v_inj_soll'],label='v_inj_soll')
# plt.plot(c1['Q_Vol_ist'],label='Q_Vol_ist')
# plt.plot(c1['V_Screw_ist'],label='V_Screw_ist')
# plt.plot(c1['p_wkz_ist'],label='p_wkz_ist')
# plt.plot(c1['T_wkz_ist'],label='T_wkz_ist')
# plt.plot(c1['p_inj_ist'],label='p_inj_ist')
# plt.legend()
# plt.xlim([0,0.26])


