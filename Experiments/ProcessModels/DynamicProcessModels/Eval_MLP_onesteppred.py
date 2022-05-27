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
from copy import deepcopy

import multiprocessing

import sys
# sys.path.insert(0, "E:\GitHub\DigitalTwinInjectionMolding")
sys.path.insert(0, 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, '/home/alexander/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, 'E:/GitHub/DigitalTwinInjectionMolding/')

from DIM.miscellaneous.PreProcessing import LoadDynamicData
from DIM.models.model_structures import MLP
from DIM.models.injection_molding import ProcessModel
from DIM.optim.param_optim import parallel_mode,series_parallel_mode
from DIM.optim.param_optim import ParallelModelTraining
from DIM.optim.common import BestFitRate



def Eval_MLP(dim_hidden):
    
    res = pkl.load(open('MLP_inj_h'+str(dim_hidden)+'_onestep_pred.pkl','rb'))
    params = res.loc[6]['params_val']
    
    # res = pkl.load(open('MLP_h10_3sub.pkl','rb'))
    # params = res.loc[11]['params_val']
    
    # params = res.loc[res['loss_val'].idxmin()][['params']][0]

    
    charges = list(range(1,3))    
 
    split = 'process'
    mode = 'process'

    # path = '/home/alexander/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
    path = 'E:/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
    # path = 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
    
    u_inj= ['v_inj_soll']
    u_press= ['p_inj_soll']
    u_cool= []
    
    u_lab = [u_inj,u_press,u_cool]
    
    # u_lab = [u_inj,[],[]]
    y_lab = ['Q_Vol_ist','V_Screw_ist','p_wkz_ist','T_wkz_ist','p_inj_ist']
    
    data_train, data_val = LoadDynamicData(path,charges,split,y_lab,u_lab,mode)

    data_inj_train = deepcopy(data_train)
    data_inj_val = deepcopy(data_val)
    
    data_press_train = deepcopy(data_train)
    data_press_val = deepcopy(data_val)
    
    data_cool_train = deepcopy(data_train)
    data_cool_val = deepcopy(data_val)
    
    
    for i in range(len(data_inj_train['data'])):
        t1 = data_inj_train['switch'][i][0]
        data_inj_train['data'][i] = data_inj_train['data'][i].loc[0:t1]
        
        t2 = data_press_train['switch'][i][1]
        data_press_train['data'][i] = data_press_train['data'][i].loc[t1:t2]
        
        data_cool_train['data'][i] = data_cool_train['data'][i].loc[t2::]
        
    
    for i in range(len(data_inj_val['data'])):
        t1 = data_inj_val['switch'][i][0]
        data_inj_val['data'][i] = data_inj_val['data'][i].loc[0:t1]
        
        t2 = data_inj_val['switch'][i][1]
        data_press_val['data'][i] = data_press_val['data'][i].loc[t1:t2]
        
        data_cool_val['data'][i] = data_cool_val['data'][i].loc[t2::]

    inj_model = MLP(dim_u=1,dim_out=5,dim_hidden=dim_hidden,u_label=u_inj,
                    y_label=y_lab,name='inj')

    inj_model.Parameters = params
    # press_model = MLP(dim_u=1,dim_out=5,dim_hidden=dim_hidden,name='press')
    # cool_model = MLP(dim_u=0,dim_out=5,dim_hidden=dim_hidden,name='cool')

    # process_model = ProcessModel(subsystems=[inj_model,press_model,cool_model],
    #                               name='p_model')   

    # Assign best parameters to model
    # process_model.SetParameters(params)
    
    # Evaluate model on training data
    # _,y_train = parallel_mode(inj_model,data_inj_train)
    _,y_train = series_parallel_mode(inj_model,data_inj_train)
    
    BFR_train = []
    
    for c in range(0,len(data_inj_train['data'])):
        idx = y_train[c].index
        y_true = np.array(data_inj_train['data'][c].loc[idx][y_lab]) 
        y_est = np.array(y_train[c])
        BFR_train.append(BestFitRate(y_true,y_est))

    BFR_train = np.array(BFR_train).reshape((-1,1))
        
    results_train = pd.DataFrame(data=np.hstack([BFR_train]),
                                index = data_inj_train['cycle_num'],
                                columns=['BFR'])
    
    # Evaluate model on validation data
    # _,y_val = parallel_mode(inj_model,data_inj_val)
    _,y_val = series_parallel_mode(inj_model,data_inj_val)
    
    BFR_val = []
    
    for c in range(0,len(data_inj_val['data'])):
        idx = y_val[c].index
        y_true = np.array(data_inj_val['data'][c].loc[idx][y_lab]) 
        y_est = np.array(y_val[c])
        BFR_val.append(BestFitRate(y_true,y_est))

    BFR_val = np.array(BFR_val).reshape((-1,1))
    
    results_val = pd.DataFrame(data=np.hstack([BFR_val]),
                                index = data_inj_val['cycle_num'],
                                columns=['BFR'])


    return results_train, results_val 

results_train, results_val = Eval_MLP(dim_hidden=10)


# # Worst batch
# cyc = results_val.index.get_loc(521)

# _,_,_,y_val = parallel_mode(process_model,[data['u_val'][cyc]],[data['y_val'][cyc]],
#                           [data['init_state_val'][cyc]],[data['switch_val'][cyc]])

# plt.figure()
# plt.plot(np.array(data['y_val'][cyc]),label=['Q_Vol','V_Screw','p_wkz',
#                                    'T_wkz','p_inj'])
# plt.plot(np.array(y_val[0]),label=['Q_Vol_est','V_Screw_est','p_wkz_est',
#                                    'T_wkz_est','p_inj_est'])
# plt.legend()


# # Best batch
# cyc = results_val.index.get_loc(1412)

# _,_,_,y_val = parallel_mode(process_model,[data['u_val'][cyc]],[data['y_val'][cyc]],
#                           [data['init_state_val'][cyc]],[data['switch_val'][cyc]])
# plt.figure()
# plt.plot(np.array(data['y_val'][cyc]),label=['Q_Vol','V_Screw','p_wkz',
#                                    'T_wkz','p_inj'])
# plt.plot(np.array(y_val[0]),label=['Q_Vol_est','V_Screw_est','p_wkz_est',
#                                    'T_wkz_est','p_inj_est'])
# plt.legend()

