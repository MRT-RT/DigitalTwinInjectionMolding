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
sys.path.insert(0, 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/')

from DIM.models.model_structures import GRU
from DIM.models.injection_molding import QualityModel
from DIM.optim.common import BestFitRate
from DIM.optim.param_optim import parallel_mode
from DIM.miscellaneous.PreProcessing import arrange_data_for_ident, eliminate_outliers, LoadDynamicData


def Eval_GRU_on_Val(dim_c):


    # Load best model
    res = pkl.load(open('GRU_c'+str(dim_c)+'_3sub_all.pkl','rb'))
       
    # params = res.loc[res['loss_val'].idxmin()][['params']][0]
    params = res.loc[0]['params_val']

    charges = list(range(1,275))
    
    split = 'all'
    # split = 'part'
    
    path = 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
    # path = '/home/alexander/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
    # path = 'E:/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
       
   
    u_inj= ['p_wkz_ist','T_wkz_ist']
    u_press= ['p_wkz_ist','T_wkz_ist']
    u_cool= ['p_wkz_ist','T_wkz_ist']
    
    u_lab = [u_inj,u_press,u_cool]
    y_lab = ['Durchmesser_innen']
    
    data,cycles_train_label,cycles_val_label,charge_train_label,charge_val_label = \
    LoadDynamicData(path,charges,split,y_lab,u_lab)
    
    c0_train = [np.zeros((dim_c,1)) for i in range(0,len(data['u_train']))]
    c0_val = [np.zeros((dim_c,1)) for i in range(0,len(data['u_val']))]    
    
    data['init_state_train'] = c0_train
    data['init_state_val'] = c0_val
    
    
    inj_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=1,dim_out=1,name='inj')
    press_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=1,dim_out=1,name='press')
    cool_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=10,dim_out=1,name='cool')
      
    quality_model = QualityModel(subsystems=[inj_model,press_model,cool_model],
                                  name='q_model')    
    
    # Assign best parameters to model
    quality_model.SetParameters(params)
    
    # Evaluate model on training data
    _,e_train,_,y_train = parallel_mode(quality_model,data['u_train'],data['y_train'],
                             data['init_state_train'],data['switch_train'])
    
    y_true = np.array(data['y_train']).reshape((-1,1))
    y_train = np.array(y_train).reshape((-1,1))
    e_train = np.array(e_train).reshape((-1,1))
    cycles_train_label = np.array(cycles_train_label).reshape((-1,))
    charge_train_label = np.array(charge_train_label).reshape((-1,1))  
    
    # print(y_true.shape,y_train.shape,e_train.shape,charge_train_label.shape)
    
    results_train = pd.DataFrame(data=np.hstack([y_true,y_train,e_train,
                                  charge_train_label]),
                                index = cycles_train_label,
                            columns=['y_true','y_est','e','charge'])
    
    # Evaluate model on validation data
    _,e_val,_,y_val = parallel_mode(quality_model,data['u_val'],data['y_val'],
                              data['init_state_val'],data['switch_val'])
    
    
    y_true = np.array(data['y_val']).reshape((-1,1))
    y_val = np.array(y_val).reshape((-1,1))
    e_val = np.array(e_val).reshape((-1,1))
    cycles_val_label = np.array(cycles_val_label).reshape((-1,))
    charge_val_label = np.array(charge_val_label).reshape((-1,1))
    
    results_val = pd.DataFrame(data=np.hstack([y_true,y_val,e_val,
                                  charge_val_label]),
                                index = cycles_val_label,
                            columns=['y_true','y_est','e','charge'])

    return results_train, results_val, data, quality_model



results_train, results_val, data, quality_model = Eval_GRU_on_Val(dim_c=4)



# pkl.dump(results_train,open('GRU_results_train_c'+str(c)+'.pkl','wb')) 
# pkl.dump(results_val,open('GRU_results_val_c'+str(c)+'.pkl','wb')) 
# pkl.dump(quality_model,open('GRU_quality_model_c'+str(c)+'.pkl','wb'))
# pkl.dump(data,open('data_c'+str(c)+'.pkl','wb'))