#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 10:17:29 2022

@author: alexander
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

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
from DIM.optim.param_optim import ModelTraining

charges = list(range(1,3))    
 
split = 'process'
mode = 'process'

# # path = '/home/alexander/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
path = 'E:/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
# # path = 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'

u_inj= ['v_inj_soll']
u_press= ['p_inj_soll']
u_cool= []

u_lab = [u_inj,u_press,u_cool]

# u_lab = [u_inj,[],[]]
y_lab = ['Q_Vol_ist','V_Screw_ist','p_wkz_ist','T_wkz_ist','p_inj_ist']

data_train, data_val = LoadDynamicData(path,charges,split,y_lab,u_lab,mode)

data_inj_train = data_train.copy()
data_inj_val = data_val.copy()

for i in range(len(data_inj_train['data'])):
    t1 = data_inj_train['switch'][i][0]
    data_inj_train['data'][i] = data_inj_train['data'][i].loc[0:t1]

for i in range(len(data_inj_val['data'])):
    t1 = data_inj_val['switch'][i][0]
    data_inj_val['data'][i] = data_inj_val['data'][i].loc[0:t1]

# data_cool = data.copy()

# data_inj['u_train'] = [u[0] for u in data_inj['u_train']]
# data_press['u_train'] = [u[0] for u in data_press['u_train']]
# data_cool['u_train'] = [u[0] for u in data_cool['u_train']]

# data_inj['u_val'] = [u[0] for u in data_inj['u_val']]
# data_press['u_val'] = [u[0] for u in data_press['u_val']]
# data_cool['u_val'] = [u[0] for u in data_cool['u_val']]



# for u_key, y_key in zip(['u_train','u_val'],['y_train','y_val']):
#     data_inj[u_key] = [el[0] for el in data_inj[u_key]] 
#     data_inj[y_key] = [data_inj[y_key][i][0:data_inj[switch_train]
                                          
dim_hidden = 10

inj_model = MLP(dim_u=1,dim_out=5,dim_hidden=dim_hidden,
                u_label=u_inj,y_label=y_lab ,name='inj')
# # press_model = MLP(dim_u=1,dim_out=5,dim_hidden=dim_hidden,name='press')
# # cool_model = MLP(dim_u=0,dim_out=5,dim_hidden=dim_hidden,name='cool')

# inj_training = ModelTraining(inj_model,data_inj_train,data_inj_val,initializations=5,mode='series')

inj_training = pkl.load(open('inj_training.pkl','rb'))

params_opt = inj_training.loc[1]['params_val']

inj_model.Parameters = params_opt

# Worst batch
cyc = 1

_,simulation = parallel_mode(inj_model,data_inj_val,params=None)

# _,simulation = series_parallel_mode(inj_model,data_inj_val,params=None)


plt.figure()
plt.plot(data_inj_val['data'][cyc][y_lab])
plt.plot(simulation[cyc])
plt.legend()

# def Fit_MLP(dim_hidden,initial_params=None):

#     charges = list(range(1,3))    
 
#     split = 'process'
#     mode = 'process'

#     # path = '/home/alexander/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
#     path = 'E:/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
#     # path = 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
    
#     u_inj= ['v_inj_soll']
#     u_press= ['p_inj_soll']
#     u_cool= []

#     u_lab = [u_inj,u_press,u_cool]
#     y_lab = ['Q_Vol_ist','V_Screw_ist','p_wkz_ist','T_wkz_ist','p_inj_ist']

#     data,cycles_train_label,cycles_val_label,charge_train_label,charge_val_label = \
#     LoadDynamicData(path,charges,split,y_lab,u_lab,mode)

#     inj_model = MLP(dim_u=1,dim_out=5,dim_hidden=dim_hidden,name='inj')
#     press_model = MLP(dim_u=1,dim_out=5,dim_hidden=dim_hidden,name='press')
#     cool_model = MLP(dim_u=0,dim_out=5,dim_hidden=dim_hidden,name='cool')

#     process_model = ProcessModel(subsystems=[inj_model,press_model,cool_model],
#                                   name='p_model')   


#     s_opts = {"max_iter": 2000, 'hessian_approximation':'limited-memory'}

#     results_MLP =  ModelTraining(process_model,data,initializations=2, 
#                     BFR=False, p_opts=None, s_opts=s_opts,mode='series')
    
#     pkl.dump(results_MLP,open('MLP_h'+str(dim_hidden)+'_3sub.pkl','wb'))


#     return results_MLP

# h10 = Fit_MLP(dim_hidden=10)

# if __name__ == '__main__':
#     multiprocessing.freeze_support()
#     # h1 = Fit_MLP(dim_hidden=1)
#     # h2 = Fit_MLP(dim_hidden=2)
#     h10 = Fit_MLP(dim_hidden=10)
#     h3 = Fit_MLP(dim_hidden=3)
#     h4 = Fit_MLP(dim_hidden=4)
#     h5 = Fit_MLP(dim_hidden=5)
#     h6 = Fit_MLP(dim_hidden=6)
#     h7 = Fit_MLP(dim_hidden=7)
#     h8 = Fit_MLP(dim_hidden=8)
#     h9 = Fit_MLP(dim_hidden=9)
#     h10 = Fit_MLP(dim_hidden=10)

