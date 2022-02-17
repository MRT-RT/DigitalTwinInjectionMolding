# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:44:37 2021

@author: alexa
"""

import pickle as pkl
import numpy as np

import multiprocessing

import sys
# sys.path.insert(0, "E:\GitHub\DigitalTwinInjectionMolding")
sys.path.insert(0, 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, '/home/alexander/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, 'E:/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, 'C:/Users/LocalAdmin/Documents/GitHub/DigitalTwinInjectionMolding/')

from DIM.miscellaneous.PreProcessing import arrange_data_for_ident, eliminate_outliers
from DIM.models.model_structures import GRU
from DIM.models.injection_molding import QualityModel
from DIM.optim.param_optim import ParallelModelTraining, ModelTraining
from DIM.miscellaneous.PreProcessing import LoadDynamicData

''' Data Loading '''  
charges = list(range(1,275))
dim_c = 1

# split = 'all'
split = 'part'

# path = 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
# path = '/home/alexander/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
# path = 'E:/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
path = 'C:/Users/LocalAdmin/Documents/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'

u_inj= ['p_wkz_ist','T_wkz_ist']
u_press= ['p_wkz_ist','T_wkz_ist']
u_cool= ['p_wkz_ist','T_wkz_ist']

u_lab = [u_inj,u_press,u_cool]
y_lab = ['Durchmesser_innen']

data,cycles_train_label,cycles_val_label,charge_train_label,charge_val_label = \
LoadDynamicData(path,charges,split,y_lab,u_lab)

c0_train = [np.ones((dim_c,1)) for i in range(0,len(data['u_train']))]
c0_val = [np.ones((dim_c,1)) for i in range(0,len(data['u_val']))]    

data['init_state_train'] = c0_train
data['init_state_val'] = c0_val
    
''' Model Initialiazion '''    
inj_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=1,dim_out=1,name='inj')
press_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=1,dim_out=1,name='press')
cool_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=10,dim_out=1,name='cool')
    
    
quality_model = QualityModel(subsystems=[inj_model,press_model,cool_model],
                              name='q_model')
    
''' Optimize injection model '''

# Freeze all other models
press_model.FrozenParameters = list(press_model.Parameters.keys())
cool_model.FrozenParameters = list(press_model.Parameters.keys())

# Initialize biases of forget gate
press_model.InitialParameters =  {'b_z_press':np.ones((1,1))*-(1e100)}
cool_model.InitialParameters =  {'b_z_cool':np.ones((1,1))*-(1e100)}



s_opts = {"max_iter": 100,'step':0.1}
    
res1 = ModelTraining(quality_model,data,initializations=1, BFR=False, 
                  p_opts=None, s_opts=s_opts)


    results_GRU['Chargen'] = 'c'+str(counter)
    
    pkl.dump(results_GRU,open('GRU_Durchmesser_innen_c'+str(counter)+'_tuned_work.pkl','wb'))
    
    print('Charge '+str(counter)+' finished.')
    
    return results_GRU  

initial_params = {'b_z_press':np.ones((1,1))*-(1e100),
                  'b_z_cool':np.ones((1,1))*-(1e100),
                  'b_r_press':np.ones((1,1))*1e100,
                  'b_c_press':np.zeros((1,1)),
                  'b_r_cool':np.zeros((1,1))*1e100,
                  'b_c_cool':np.zeros((1,1))}

result = Fit_GRU(44,initial_params)



