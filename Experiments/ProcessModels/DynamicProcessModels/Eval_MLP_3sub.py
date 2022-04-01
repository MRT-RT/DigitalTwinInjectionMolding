#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 10:17:29 2022

@author: alexander
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import pandas as pd

import multiprocessing

import sys
# sys.path.insert(0, "E:\GitHub\DigitalTwinInjectionMolding")
sys.path.insert(0, 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, '/home/alexander/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, 'E:/GitHub/DigitalTwinInjectionMolding/')

from DIM.miscellaneous.PreProcessing import LoadDynamicData
from DIM.models.model_structures import MLP
from DIM.models.injection_molding import ProcessModel
from DIM.optim.param_optim import parallel_mode
from DIM.optim.param_optim import ParallelModelTraining
from DIM.optim.common import BestFitRate



def Eval_MLP(dim_hidden):
    
    res = pkl.load(open('MLP_h'+str(dim_hidden)+'_3sub.pkl','rb'))
    
    # params = res.loc[res['loss_val'].idxmin()][['params']][0]
    params = res.loc[11]['params_val']
    
    charges = list(range(1,275))    
 
    split = 'process'
    mode = 'process'

    # path = '/home/alexander/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
    path = 'E:/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
    # path = 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
    
    u_inj= ['v_inj_soll']
    u_press= ['p_inj_soll']
    u_cool= []

    u_lab = [u_inj,u_press,u_cool]
    y_lab = ['Q_Vol_ist','V_Screw_ist','p_wkz_ist','T_wkz_ist','p_inj_ist']

    data,cycles_train_label,cycles_val_label,charge_train_label,charge_val_label = \
    LoadDynamicData(path,charges,split,y_lab,u_lab,mode)

    inj_model = MLP(dim_u=1,dim_out=5,dim_hidden=dim_hidden,name='inj')
    press_model = MLP(dim_u=1,dim_out=5,dim_hidden=dim_hidden,name='press')
    cool_model = MLP(dim_u=0,dim_out=5,dim_hidden=dim_hidden,name='cool')

    process_model = ProcessModel(subsystems=[inj_model,press_model,cool_model],
                                  name='p_model')   

    # Assign best parameters to model
    process_model.SetParameters(params)
    
    # Evaluate model on training data
    _,_,_,y_train = parallel_mode(process_model,data['u_train'],data['y_train'],
                              data['init_state_train'],data['switch_train'])
    
    BFR_train = []
    
    for i in range(0,len(data['y_train'])):
        y_true = np.array(data['y_train'][i]) 
        y_est = np.array(y_train[i])
        BFR_train.append(BestFitRate(y_true,y_est))

    BFR_train = np.array(BFR_train).reshape((-1,1))
    
    cycles_train_label = np.array(cycles_train_label).reshape((-1,))
    charge_train_label = np.array(charge_train_label).reshape((-1,1))  
    
    # print(y_true.shape,y_train.shape,e_train.shape,charge_train_label.shape)
    
    results_train = pd.DataFrame(data=np.hstack([BFR_train,
                                  charge_train_label]),
                                index = cycles_train_label,
                            columns=['BFR','charge'])
    
    # Evaluate model on validation data
    _,_,_,y_val = parallel_mode(process_model,data['u_val'],data['y_val'],
                              data['init_state_val'],data['switch_val'])

    BFR_val = []
    
    for i in range(0,len(data['y_val'])):
        y_true = np.array(data['y_val'][i]) 
        y_est = np.array(y_val[i])
        BFR_val.append(BestFitRate(y_true,y_est))

    BFR_val = np.array(BFR_val).reshape((-1,1))
    
    cycles_val_label = np.array(cycles_val_label).reshape((-1,))
    charge_val_label = np.array(charge_val_label).reshape((-1,1))
    
    results_val = pd.DataFrame(data=np.hstack([BFR_val,
                                  charge_val_label]),
                                index = cycles_val_label,
                            columns=['BFR','charge'])


    return results_train, results_val, process_model, data

results_train, results_val, process_model, data = Eval_MLP(dim_hidden=10)

# Worst batch
cyc = results_val.index.get_loc(521)

_,_,_,y_val = parallel_mode(process_model,[data['u_val'][cyc]],[data['y_val'][cyc]],
                          [data['init_state_val'][cyc]],[data['switch_val'][cyc]])

plt.figure()
plt.plot(np.array(data['y_val'][cyc]),label=['Q_Vol','V_Screw','p_wkz',
                                   'T_wkz','p_inj'])
plt.plot(np.array(y_val[0]),label=['Q_Vol_est','V_Screw_est','p_wkz_est',
                                   'T_wkz_est','p_inj_est'])
plt.legend()


# Best batch
cyc = results_val.index.get_loc(1412)

_,_,_,y_val = parallel_mode(process_model,[data['u_val'][cyc]],[data['y_val'][cyc]],
                          [data['init_state_val'][cyc]],[data['switch_val'][cyc]])
plt.figure()
plt.plot(np.array(data['y_val'][cyc]),label=['Q_Vol','V_Screw','p_wkz',
                                   'T_wkz','p_inj'])
plt.plot(np.array(y_val[0]),label=['Q_Vol_est','V_Screw_est','p_wkz_est',
                                   'T_wkz_est','p_inj_est'])
plt.legend()

