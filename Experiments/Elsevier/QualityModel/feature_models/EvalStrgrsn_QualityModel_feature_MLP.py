# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 15:16:22 2022

@author: LocalAdmin
"""

import sys
sys.path.insert(0, '/home/alexander/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, 'E:/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/')

# import os.path as path
# two_up =  path.abspath(path.join(__file__ ,"../.."))
# print(two_up)

from DIM.miscellaneous.PreProcessing import LoadFeatureData, MinMaxScale
from DIM.optim.common import BestFitRate
from DIM.models.model_structures import Static_MLP
from DIM.optim.param_optim import ParallelModelTraining, static_mode

import multiprocessing

import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def Eval_MLP(dim_hidden):
    
    res = pkl.load(open('QualityModel_Gewicht_feature_MLP_'+str(dim_hidden)+'.pkl','rb'))
    params = res.loc[res['loss_val'].idxmin()][['params_val']][0]
    
    charges = list(range(1,26)) # list(range(1,26))
    
    split = 'all'
    
    # path_sys = 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/'  
    path_sys = '/home/alexander/GitHub/DigitalTwinInjectionMolding/'
    # path_sys = 'E:/GitHub/DigitalTwinInjectionMolding/'
    
    path_data_train = 'data/Stoergroessen/20220504/Versuchsplan/normalized/'
    
    # path_data_strgrsn = 'data/Stoergroessen/20220506/Rezyklat_Stoerung/normalized/'
    path_data_strgrsn = 'data/Stoergroessen/20220504/Umschaltpkt_Stoerung/normalized/'
    # path_data_strgrsn = 'data/Stoergroessen/20220505/T_wkz_Stoerung/normalized/'
    
    # path = '/home/alexander/GitHub/DigitalTwinInjectionMolding/data/Stoergroessen/20220504/Versuchsplan/normalized/'
    # path = 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/data/Stoergroessen/20220504/Versuchsplan/normalized/'
    
    # path = 'E:/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
    
    
    
    data_train,_ = LoadFeatureData(path_sys+path_data_train,charges,split)
    data_st1,data_st2 = LoadFeatureData(path_sys+path_data_strgrsn,[1],split)
    
    data_st = pd.concat([data_st1,data_st2])
    
    u_label = ['T_wkz_0', 'T_wkz_max', 't_Twkz_max', 'T_wkz_int', 'p_wkz_max',
    'p_wkz_int', 'p_wkz_res', 't_pwkz_max']
    
    y_label = ['Gewicht']   
    
    # Normalize Data
    data_train,minmax = MinMaxScale(data_train,u_label+y_label)
    data_st,_ = MinMaxScale(data_st,u_label+y_label,minmax)
    
    model = Static_MLP(dim_u=8, dim_out=1, dim_hidden=dim_hidden,u_label=u_label,
                        y_label=y_label,name='MLP', init_proc='xavier')
    

    # Assign best parameters to model
    model.Parameters = params
    
    # Evaluate model on training data
    _,y_train = static_mode(model,data_train)
    y_true = data_train[y_label].values.reshape((-1,1))
    e_train = y_true-y_train
    
    results_train = pd.DataFrame(data=np.hstack([y_true,y_train,e_train]),
                            columns=['y_true','y_est','e'],
                              index = data_train.index)
    
    # Evaluate model on validation data
    _,y_val = static_mode(model,data_st)
        
    y_true = data_st[y_label].values.reshape((-1,1))
    e_val = y_true-y_val
    
    results_val = pd.DataFrame(data=np.hstack([y_true,y_val,e_val]),
                            columns=['y_true','y_est','e'],
                            index = data_st.index)

    return results_train,results_val

for i in range(1,11):
    
    results_train,results_val = Eval_MLP(dim_hidden=i)
    
    plt.figure()
    plt.plot(results_val['y_true'],'o')
    plt.plot(results_val['y_est'],'o')
    plt.title('MLP ' + str(i))
    
    print(BestFitRate(results_val['y_true'].values.reshape((-1,1)),
                results_val['y_est'].values.reshape((-1,1))))









