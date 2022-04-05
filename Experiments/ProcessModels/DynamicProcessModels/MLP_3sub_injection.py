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

path = '/home/alexander/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
# path = 'E:/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
# # path = 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'

u_inj= ['v_inj_soll']
u_press= ['p_inj_soll']
u_cool= []

u_lab = [u_inj,u_press,u_cool]

# u_lab = [u_inj,[],[]]
y_lab = ['Q_Vol_ist','V_Screw_ist','p_wkz_ist','T_wkz_ist','p_inj_ist']

data,cycles_train_label,cycles_val_label,charge_train_label,charge_val_label = \
LoadDynamicData(path,charges,split,y_lab,u_lab,mode)

data_inj = data.copy()
data_press = data.copy()
data_cool = data.copy()

for u_key, y_key in zip(['u_train','u_val'],['y_train','y_val']):
    data_inj[u_key] = [el[0] for el in data_inj[u_key]] 
    data_inj[y_key] = [data_inj[y_key][i][0:len(data_inj[u_key][i])] for i in range(0,len(data_inj[u_key]))]

dim_hidden = 10

inj_model = MLP(dim_u=1,dim_out=5,dim_hidden=dim_hidden,name='inj')
# # press_model = MLP(dim_u=1,dim_out=5,dim_hidden=dim_hidden,name='press')
# # cool_model = MLP(dim_u=0,dim_out=5,dim_hidden=dim_hidden,name='cool')

# inj_training = ModelTraining(inj_model,data_inj,initializations=10,mode='series')

inj_training = pkl.load(open('inj_training.pkl','rb'))

params_opt = inj_training.loc[5]['params_val']

inj_model.Parameters = params_opt

# Worst batch
cyc = 1

_,_,_,y_val = parallel_mode(inj_model,[data_inj['u_val'][cyc]],[data_inj['y_val'][cyc]],
                          [data_inj['init_state_val'][cyc]],[data_inj['switch_val'][cyc]])

# _,_,y_val = series_parallel_mode(inj_model,[data_inj['u_val'][cyc]],None,
#                                    [data_inj['y_val'][cyc]],
#                                    [data_inj['init_state_val'][cyc]])


plt.figure()
plt.plot(np.array(data_inj['y_val'][cyc][1:,:]),label=['Q_Vol','V_Screw','p_wkz',
                                    'T_wkz','p_inj'])
plt.plot(np.array(y_val[0]),label=['Q_Vol_est','V_Screw_est','p_wkz_est',
                                    'T_wkz_est','p_inj_est'])
plt.legend()

# def Fit_MLP(dim_hidden,initial_params=None):

#     charges = list(range(1,275))    
 
#     split = 'process'
#     mode = 'process'

#     # path = '/home/alexander/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
#     # path = 'E:/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
#     path = 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
    
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

#     results_MLP =  ParallelModelTraining(process_model,data,initializations=20, 
#                     BFR=False, p_opts=None, s_opts=s_opts,mode='parallel',
#                     n_pool=8)
    
#     pkl.dump(results_MLP,open('MLP_h'+str(dim_hidden)+'_3sub.pkl','wb'))


#     return results_MLP

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

