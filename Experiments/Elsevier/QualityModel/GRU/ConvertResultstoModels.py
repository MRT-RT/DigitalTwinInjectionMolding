#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 09:30:44 2022

@author: alexander
"""


import pickle as pkl
import numpy as np
import pandas as pd


import multiprocessing

import sys
# sys.path.insert(0, "E:\GitHub\DigitalTwinInjectionMolding")
sys.path.insert(0, 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, '/home/alexander/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, 'E:/GitHub/DigitalTwinInjectionMolding/')


from DIM.miscellaneous.PreProcessing import arrange_data_for_ident, eliminate_outliers
from DIM.models.model_structures import GRU
from DIM.models.injection_molding import QualityModel
from DIM.optim.param_optim import ModelTraining
from DIM.miscellaneous.PreProcessing import LoadDynamicData



dim_c = int(input('Please enter the dimension of the cell state:   '))

#### Load Results ##### #######################################################

res = pkl.load(open('InitialModels/GRU_c'+str(dim_c)+'_3sub_all.pkl','rb'))


for i in res.index:
    res.loc[i,'loss_val'] = float(res.loc[i,'loss_val'])
    res.loc[i,'loss_train'] = float(res.loc[i,'loss_train'])
    
res['loss_val'] = pd.to_numeric(res['loss_val'])
res['loss_train'] = pd.to_numeric(res['loss_train'])

pkl.dump(res,open('GRU_c'+str(dim_c)+'_3sub_all.pkl','wb'))

#### Parameterize Model #######################################################

u_inj= ['p_wkz_ist','T_wkz_ist']
u_press= ['p_wkz_ist','T_wkz_ist']
u_cool= ['p_wkz_ist','T_wkz_ist']

u_lab = [u_inj,u_press,u_cool]
y_lab = ['Durchmesser_innen']


inj_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=1,
                u_label=u_inj,y_label=y_lab,dim_out=1,name='inj')

press_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=1,
                  u_label=u_press,y_label=y_lab,dim_out=1,name='press')

cool_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=10,
                 u_label=u_cool,y_label=y_lab,dim_out=1,name='cool')

quality_model = QualityModel(subsystems=[inj_model,press_model,cool_model],
                              name='q_model')


res_dict = res.to_dict()


res_dict['model_val'] = {}
res_dict['model_train'] = {}

for key in res_dict['params_val'].keys():
    
    inj_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=1,
                    u_label=u_inj,y_label=y_lab,dim_out=1,name='inj')

    press_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=1,
                      u_label=u_press,y_label=y_lab,dim_out=1,name='press')

    cool_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=10,
                     u_label=u_cool,y_label=y_lab,dim_out=1,name='cool')

    quality_model = QualityModel(subsystems=[inj_model,press_model,cool_model],
                                  name='q_model')
    
    
    quality_model.SetParameters(res_dict['params_val'][key])
    
    res_dict['model_val'][key] = quality_model 


for key in res_dict['params_train'].keys():
    
    inj_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=1,
                    u_label=u_inj,y_label=y_lab,dim_out=1,name='inj')

    press_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=1,
                      u_label=u_press,y_label=y_lab,dim_out=1,name='press')

    cool_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=10,
                     u_label=u_cool,y_label=y_lab,dim_out=1,name='cool')

    quality_model = QualityModel(subsystems=[inj_model,press_model,cool_model],
                                  name='q_model')
    
    
    quality_model.SetParameters(res_dict['params_train'][key])
    res_dict['model_train'][key] = quality_model 
    
    
pkl.dump(res_dict,open('GRU_c'+str(dim_c)+'_3sub_all.dict','wb'))
# quality_model.SetParameters()


