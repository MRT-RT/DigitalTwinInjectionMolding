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
from DIM.optim.param_optim import ModelTraining





def Fit_MLP(dim_c,dim_hidden,initial_params=None):

    charges = list(range(1,2))  
 
    split = 'process'
    mode = 'process'
    
    path = '/home/alexander/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/normalized/'
    # path = 'E:/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
    # path = 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/normalized/'
    
    u_inj= ['v_inj_soll']
    u_press= ['p_inj_soll']
    u_cool= []
    
    u_lab = [u_inj,u_press,u_cool]
    
    # u_lab = [u_inj,[],[]]
    y_lab = ['Q_Vol_ist','V_Screw_ist','p_wkz_ist','T_wkz_ist','p_inj_ist']
    
    data_train, data_val = LoadDynamicData(path,charges,split,y_lab,u_lab,mode,None)
    
    
    
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
        data_press_train['init_state'][i] = data_press_train['data'][i][y_lab].loc[t1].values.reshape((5,1))
        
        
        data_cool_train['data'][i] = data_cool_train['data'][i].loc[t2::]
        data_cool_train['init_state'][i] = data_cool_train['data'][i][y_lab].loc[t2].values.reshape((5,1))
    
    for i in range(len(data_inj_val['data'])):
        t1 = data_inj_val['switch'][i][0]
        data_inj_val['data'][i] = data_inj_val['data'][i].loc[0:t1]
        
        t2 = data_inj_val['switch'][i][1]
        data_press_val['data'][i] = data_press_val['data'][i].loc[t1:t2]
        data_press_val['init_state'][i] = data_press_val['data'][i][y_lab].loc[t1].values.reshape((5,1))
        
        data_cool_val['data'][i] = data_cool_val['data'][i].loc[t2::]
        data_cool_val['init_state'][i] = data_cool_val['data'][i][y_lab].loc[t2].values.reshape((5,1))
    
    results_inj = []
    
    s_opts = {"max_iter": 3000, 'hessian_approximation':'limited-memory'}
    
       
    inj_model = State_MLP(dim_u=1,dim_c=dim_c,dim_hidden=dim_hidden,dim_out=5,
                          u_label=u_inj,y_label=y_lab,name='inj')
    
    C = np.zeros((5,7))
    C[[0,1,2,3,4],[0,1,2,3,4]] = 1
    
    inj_model.InitialParameters = {'C_inj':C}
    inj_model.FrozenParameters = ['C_inj']
    
    # press_model = MLP(dim_u=1,dim_c=7,dim_hidden=15,dim_out=5,u_label=u_press,
    #                 y_label=y_lab, name='press')
    
    # cool_model = MLP(dim_u=1,dim_c=7,dim_hidden=15,dim_out=5,u_label=u_cool,
    #                 y_label=y_lab,name='cool')
    

    ''' Parameter Estimation '''    
    results_inj = ModelTraining(inj_model,data_inj_train,data_inj_val,
                            initializations=10,BFR=False, p_opts=None, 
                            s_opts=s_opts,mode='parallel')
    pkl.dump(results_inj,open('MLP_inj_sim_c'+str(dim_hidden)+'_h'+str(dim_hidden)+'.pkl','wb'))
    
    # init_press = pkl.load(open('MLP_press_h'+str(dim_h)+'_onestep_pred.pkl','rb'))
    # init_press['loss_val'] = pd.to_numeric(init_press['loss_val'])
    # idx = init_press['loss_val'].idxmin()
    
    # press_model.InitialParameters = init_press['params_val'].loc[idx]
    
    # results_press = ModelTraining(press_model,data_press_train,data_press_val,
    #                         initializations=10,BFR=False, p_opts=None, 
    #                         s_opts=s_opts,mode='parallel')    
    # pkl.dump(results_press,open('MLP_press_sim_h'+str(dim_hidden)+'.pkl','wb'))
    
    # init_cool = pkl.load(open('MLP_cool_h'+str(dim_h)+'_onestep_pred.pkl','rb'))
    # init_cool['loss_val'] = pd.to_numeric(init_cool['loss_val'])
    # idx = init_cool['loss_val'].idxmin()

    # results_cool = ModelTraining(cool_model,data_cool_train,data_cool_val,
    #                         initializations=10,BFR=False, p_opts=None, 
    #                         s_opts=s_opts,mode='parallel')
    # pkl.dump(results_cool,open('MLP_cool_sim_h'+str(dim_hidden)+'.pkl','wb'))
    
    # s_opts = {"max_iter": 3000, 'hessian_approximation':'limited-memory'}

    # results_inj =  ParallelModelTraining(inj_model,data_inj_train,data_inj_val,
    #                        initializations=20,BFR=False, p_opts=None, 
    #                        s_opts=None,mode='parallel',n_pool=2)
    
    # results_inj = pd.concat(results_inj)
    # results_press = pd.concat(results_press)
    # results_cool = pd.concat(results_cool)
    
    
    
    

    return None

# h10 = Fit_MLP(dim_hidden=10)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    Fit_MLP(dim_c=7,dim_hidden=15)
    Fit_MLP(dim_hidden=10)
    Fit_MLP(dim_hidden=15)
    Fit_MLP(dim_hidden=20)
# h = pkl.load(open('MLP_inj_h30_onestep_pred.pkl','rb'))
# for i in h.index:
#     h['loss_val'].loc[i] = float(h['loss_val'].loc[i])
#     h['loss_train'].loc[i] = float(h['loss_train'].loc[i])
# pkl.dump(h,open('MLP_inj_h30_onestep_pred.pkl','wb'))