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

from DIM.models.model_structures import TimeDelay_MLP
from DIM.models.injection_molding import QualityModel
from DIM.optim.common import BestFitRate
from DIM.optim.param_optim import parallel_mode
from DIM.miscellaneous.PreProcessing import arrange_data_for_ident, eliminate_outliers, LoadDynamicData


def Eval_TDMLP(order,dim_h,initial_params=None):


    charges = list(range(1,26))
    
    mode='quality'
    split = 'all'
    
    # path = 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/data/Stoergroessen/20220504/Versuchsplan/normalized/'
    path = '/home/alexander/GitHub/DigitalTwinInjectionMolding/data/Stoergroessen/20220504/Versuchsplan/normalized/'
        
   
    u_inj= ['p_wkz_ist','T_wkz_ist']
    u_press= ['p_wkz_ist','T_wkz_ist']
    u_cool= ['p_wkz_ist','T_wkz_ist']
    
    u_lab = [u_inj,u_press,u_cool]
    y_lab = ['Gewicht']
    
    u_lab = [u_inj,u_press,u_cool]
    y_lab = ['Gewicht']
    
    data_train,data_val = \
    LoadDynamicData(path,charges,split,y_lab,u_lab,mode)
    
    c0_train = [np.zeros((order,1)) for i in range(0,len(data_train['data']))]
    c0_val = [np.zeros((order,1)) for i in range(0,len(data_val['data']))] 
    
    data_train['init_state'] = c0_train
    data_val['init_state'] = c0_val
    
    
    inj_model = TimeDelay_MLP(dim_u=2,dim_hidden=dim_h,dim_out=1,dim_c=order,
                    u_label=u_inj,y_label=y_lab,name='inj')
    
    press_model = TimeDelay_MLP(dim_u=2,dim_hidden=dim_h,dim_out=1,dim_c=order,
                    u_label=u_inj,y_label=y_lab,name='press')
    
    cool_model = TimeDelay_MLP(dim_u=2,dim_hidden=dim_h,dim_out=1,dim_c=order,
                    u_label=u_inj,y_label=y_lab,name='cool')
       
    quality_model = QualityModel(subsystems=[inj_model,press_model,cool_model],
                                  name='q_model')  
    
    
    # Evaluate model on training data
    _,y_train = parallel_mode(quality_model,data_train)
    y_true = np.array([df[y_lab].iloc[0] for df in data_train['data']]).reshape((-1,1))
    y_train = np.array([df[y_lab].iloc[0] for df in y_train]).reshape((-1,1))
    e_train = y_true-y_train
    
    results_train = pd.DataFrame(data=np.hstack([y_true,y_train,e_train]),
                            columns=['y_true','y_est','e'],
                              index = data_train['cycle_num'])
    


    return results_train

results_train,results_st = Eval_TDMLP(order=2,dim_h=1)


# for i in range(2,3):

#     results_train,results_st = Eval_GRU_on_Val(dim_c=i)
    
#     print(BestFitRate(results_st['y_true'].values.reshape((-1,1)),
#                 results_st['y_est'].values.reshape((-1,1))))
