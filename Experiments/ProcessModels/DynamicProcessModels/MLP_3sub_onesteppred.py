#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 10:17:29 2022

@author: alexander
"""

# import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
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
from DIM.optim.param_optim import parallel_mode, series_parallel_mode
from DIM.optim.param_optim import ParallelModelTraining

def Fit_MLP(dim_hidden,initial_params=None):

    charges = list(range(1,275))    
 
    split = 'process'
    mode = 'process'
    
    # path = '/home/alexander/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
    # path = 'E:/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
    path = 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
    
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
    
    # press_model = MLP(dim_u=1,dim_out=5,dim_hidden=dim_hidden,name='press')
    # cool_model = MLP(dim_u=0,dim_out=5,dim_hidden=dim_hidden,name='cool')

    # s_opts = {"max_iter": 3000, 'hessian_approximation':'limited-memory'}

    results_inj =  ParallelModelTraining(inj_model,data_inj_train,data_inj_val,
                           initializations=20,BFR=False, p_opts=None, 
                           s_opts=None,mode='series',n_pool=2)
    
    pkl.dump(results_inj,open('MLP_inj_h'+str(dim_hidden)+'_onestep_pred.pkl','wb'))


    return results_inj

# h10 = Fit_MLP(dim_hidden=10)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    # h5 = Fit_MLP(dim_hidden=5)
    # h10 = Fit_MLP(dim_hidden=10)
    # h15 = Fit_MLP(dim_hidden=15)
    # h20 = Fit_MLP(dim_hidden=20)
    # h25 = Fit_MLP(dim_hidden=25)
    h30 = Fit_MLP(dim_hidden=30)
    h35 = Fit_MLP(dim_hidden=35)
    h40 = Fit_MLP(dim_hidden=40)
    h50 = Fit_MLP(dim_hidden=50)

