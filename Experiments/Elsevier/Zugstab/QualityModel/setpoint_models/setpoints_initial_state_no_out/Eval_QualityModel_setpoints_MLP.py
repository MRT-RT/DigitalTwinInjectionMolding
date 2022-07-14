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


def Eval_MLP(dim_hidden):
    
    res = pkl.load(open('QualityModel_MaxSp_static_MLP_'+str(dim_hidden)+'.pkl','rb'))
    params = res.loc[res['loss_val'].idxmin()][['params_val']][0]
    
    charges = list(range(1,26)) # list(range(1,26))
    
    split = 'all'
    
    # path_sys = 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/'
    # path_sys = '/home/alexander/GitHub/DigitalTwinInjectionMolding/' 
    path_sys = 'E:/GitHub/DigitalTwinInjectionMolding/'
    
    path = path_sys + '/data/Zugstab/data/normalized_minmax/'
    
    data_train,data_val = LoadFeatureData(path,charges,split)
    
    u_label = ['Düsentemperatur', 'Werkzeugtemperatur',
               'Einspritzgeschwindigkeit','Umschaltpunkt',
               'T_wkz_0','p_inj_0','x_0']
    
    # y_label = ['E-Modul']
    y_label = ['Maximalspannung'] 
    
    # Normalize Data
    data_train,minmax = MinMaxScale(data_train,u_label+y_label)
    data_val,_ = MinMaxScale(data_val,u_label+y_label,minmax)
    
    model = Static_MLP(dim_u=7, dim_out=1, dim_hidden=dim_hidden,u_label=u_label,
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
    _,y_val = static_mode(model,data_val)
        
    y_true = data_val[y_label].values.reshape((-1,1))
    e_val = y_true-y_val
    
    results_val = pd.DataFrame(data=np.hstack([y_true,y_val,e_val]),
                            columns=['y_true','y_est','e'],
                            index = data_val.index)

    return results_train,results_val


for i in list(range(1,11))+[20,40]:
    
    results_train,results_val = Eval_MLP(dim_hidden=i)
    
    print(BestFitRate(results_val['y_true'].values.reshape((-1,1)),
                results_val['y_est'].values.reshape((-1,1))))









