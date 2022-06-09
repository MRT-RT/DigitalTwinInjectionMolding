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
import pandas as pd

import multiprocessing

import sys
# sys.path.insert(0, "E:\GitHub\DigitalTwinInjectionMolding")
sys.path.insert(0, 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, '/home/alexander/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, 'E:/GitHub/DigitalTwinInjectionMolding/')

from DIM.miscellaneous.PreProcessing import LoadDynamicData
from DIM.models.model_structures import State_MLP
from DIM.models.injection_molding import ProcessModel
from DIM.optim.param_optim import parallel_mode, series_parallel_mode
from DIM.optim.param_optim import ParallelModelTraining, ModelTraining


def Fit_MLP(dim_c,dim_hidden,initial_params=None):

    charges = list(range(1,26))  
 
    split = 'all'
    mode = 'process'
    
    path_sys = 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding'
    # path_sys = '/home/alexander/GitHub/DigitalTwinInjectionMolding'
    
    path_data = '/data/Stoergroessen/20220504/Versuchsplan/normalized/'
    
    
    u_inj= ['v_inj_soll']
    u_press= ['p_inj_soll']
    u_cool= []
    
    u_lab = [u_inj,u_press,u_cool]

    y_lab = ['Q_Vol_ist','V_Screw_ist','p_wkz_ist','T_wkz_ist','p_inj_ist']
    
    data_train, data_val = LoadDynamicData(path_sys+path_data,charges,split,
                                           y_lab,u_lab,mode)
    
    for i in range(len(data_train['data'])):
        data_train['init_state'][i] = np.vstack([data_train\
                                         ['init_state'][i],np.zeros((dim_c-5,1))])

    
    for i in range(len(data_val['data'])):
        data_val['init_state'][i] = np.vstack([data_val\
                                         ['init_state'][i],np.zeros((dim_c-5,1))])

       
    s_opts = {"max_iter": 3000, 'hessian_approximation':'limited-memory'}
    
    inj_model = State_MLP(dim_u=1,dim_c=dim_c,dim_hidden=dim_hidden,dim_out=5,
                          u_label=u_inj,y_label=y_lab,name='inj')
    
    C = np.zeros((5,dim_c))
    C[[0,1,2,3,4],[0,1,2,3,4]] = 1
    
    inj_model.InitialParameters = {'C_inj':C}
    inj_model.FrozenParameters = ['C_inj']
    
    press_model = State_MLP(dim_u=1,dim_c=dim_c,dim_hidden=dim_hidden,dim_out=5,
                      u_label=u_press,y_label=y_lab, name='press')
    
    press_model.InitialParameters = {'C_press':C}
    press_model.FrozenParameters = ['C_press'] 

    
    model_p = ProcessModel([inj_model, press_model], 'proc')
    
    ''' Parameter Estimation '''    
    results = ParallelModelTraining(model_p,data_train,data_val,
                            initializations=20,BFR=False, p_opts=None, 
                            s_opts=s_opts,mode='parallel',n_pool=10)
    

    pkl.dump(results,open('joint/MLP_joint_sim_c'+str(dim_hidden)+'_h'+str(dim_hidden)+'.pkl','wb'))

    return results

# h10 = Fit_MLP(dim_hidden=10)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    c7h5 = Fit_MLP(dim_c=7,dim_hidden=5)
    c7h10 = Fit_MLP(dim_c=7,dim_hidden=10)
    c7h20 = Fit_MLP(dim_c=7,dim_hidden=20)
    c7h40 = Fit_MLP(dim_c=7,dim_hidden=40)
    
    c8h5 = Fit_MLP(dim_c=8,dim_hidden=5)
    c8h10 = Fit_MLP(dim_c=8,dim_hidden=10)
    c8h20 = Fit_MLP(dim_c=8,dim_hidden=20)
    c8h40 = Fit_MLP(dim_c=8,dim_hidden=40)
    # inj_c7h80, press_c7h80 = Fit_MLP(dim_c=7,dim_hidden=80)
    # inj_c7h100, press_c7h100 = Fit_MLP(dim_c=7,dim_hidden=100)
    
