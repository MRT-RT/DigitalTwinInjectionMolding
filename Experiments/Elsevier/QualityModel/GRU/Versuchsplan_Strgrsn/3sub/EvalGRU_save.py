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
sys.path.insert(0, 'C:/Users/LocalAdmin/Documents/GitHub/DigitalTwinInjectionMolding/')

from DIM.models.model_structures import GRU
from DIM.models.injection_molding import QualityModel
from DIM.optim.common import BestFitRate
from DIM.optim.param_optim import parallel_mode
from DIM.miscellaneous.PreProcessing import arrange_data_for_ident, eliminate_outliers, LoadDynamicData


def Eval_GRU_on_Val(dim_c):

    # Load best model
    res = pkl.load(open('GRU_c'+str(dim_c)+'_3sub_Stoergrsn_Gewicht.pkl','rb'))
    
    params = res.loc[res['loss_val'].idxmin()][['params_val']][0]
    # params = res.loc[10]['params_val']

    charges = list(range(1,26))
    
    mode='quality'
    split = 'all'
    del_outl = True
    # path_sys = 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/'
    path_sys = 'C:/Users/LocalAdmin/Documents/GitHub/DigitalTwinInjectionMolding/'
    # path_sys = '/home/alexander/GitHub/DigitalTwinInjectionMolding/' 
    # path_sys = 'E:/GitHub/DigitalTwinInjectionMolding/'
    
    path = path_sys + 'data/Stoergroessen/20220504/Versuchsplan/normalized/'      
   
    u_inj= ['p_wkz_ist','T_wkz_ist']
    u_press= ['p_wkz_ist','T_wkz_ist']
    u_cool= ['p_wkz_ist','T_wkz_ist']
    
    u_lab = [u_inj,u_press,u_cool]
    y_lab = ['Gewicht']
    
    u_lab = [u_inj,u_press,u_cool]
    y_lab = ['Gewicht']
    
    data_train,data_val = \
    LoadDynamicData(path,charges,split,y_lab,u_lab,mode,del_outl)
    
    c0_train = [np.zeros((dim_c,1)) for i in range(0,len(data_train['data']))]
    c0_val = [np.zeros((dim_c,1)) for i in range(0,len(data_val['data']))] 
    
    data_train['init_state'] = c0_train
    data_val['init_state'] = c0_val
    
    
    inj_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=1,
                    u_label=u_inj,y_label=y_lab,dim_out=1,name='inj')
    
    press_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=1,
                      u_label=u_press,y_label=y_lab,dim_out=1,name='press')
    
    cool_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=10,
                     u_label=u_cool,y_label=y_lab,dim_out=1,name='cool')
      
    quality_model = QualityModel(subsystems=[inj_model,press_model,cool_model],
                                  name='q_model')    
    
    # Assign best parameters to model
    quality_model.SetParameters(params)
    
    # Evaluate model on training data
    _,y_train = parallel_mode(quality_model,data_train)
    y_true = np.array([df[y_lab].iloc[0] for df in data_train['data']]).reshape((-1,1))
    y_train = np.array([df[y_lab].iloc[0] for df in y_train]).reshape((-1,1))
    e_train = y_true-y_train
    
    results_train = pd.DataFrame(data=np.hstack([y_true,y_train,e_train]),
                            columns=['y_true','y_est','e'],
                              index = data_train['cycle_num'])
    
    # Evaluate model on validation data
    _,y_val = parallel_mode(quality_model,data_val)
    
    
    y_true = np.array([df[y_lab].iloc[0] for df in data_val['data']]).reshape((-1,1))
    y_val = np.array([df[y_lab].iloc[0] for df in y_val]).reshape((-1,1))
    e_val = y_true-y_val
    
    results_val = pd.DataFrame(data=np.hstack([y_true,y_val,e_val]),
                            columns=['y_true','y_est','e'],
                            index = data_val['cycle_num'])

    return results_train,results_val


for i in range(1,11):

    results_train,results_st = Eval_GRU_on_Val(dim_c=i)
    
    print(BestFitRate(results_st['y_true'].values.reshape((-1,1)),
                results_st['y_est'].values.reshape((-1,1))))
