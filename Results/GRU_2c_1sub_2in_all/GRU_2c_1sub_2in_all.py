# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:44:37 2021

@author: alexa
"""

import pickle as pkl
import numpy as np

import sys
# sys.path.insert(0, "E:\GitHub\DigitalTwinInjectionMolding")

sys.path.insert(0, 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/')


from DIM.miscellaneous.PreProcessing import arrange_data_for_ident, eliminate_outliers
from DIM.models.model_structures import GRU
from DIM.models.injection_molding import QualityModel
from DIM.optim.param_optim import ModelTraining, HyperParameterPSO
from DIM.miscellaneous.PreProcessing import LoadDynamicData


def Fit_GRU_to_Charges(charges,counter):
    
    dim_c = 2
    
    path = 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
    
    u_lab= ['p_wkz_ist','T_wkz_ist']
    u_lab = [u_lab]
    
    y_lab = ['Durchmesser_innen']
    
    data,cycles_train_label,cycles_val_label,charge_train_label,charge_val_label = \
    LoadDynamicData(path,charges,y_lab,u_lab)
    
    c0_train = [np.zeros((dim_c,1)) for i in range(0,len(data['u_train']))]
    c0_val = [np.zeros((dim_c,1)) for i in range(0,len(data['u_val']))]    
    
    data['init_state_train'] = c0_train
    data['init_state_val'] = c0_val
    
    
    one_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=5,dim_out=1,name='one_model')

    
    for rnn in [one_model]:
        name = rnn.name
        
        initial_params = {'b_r_'+name: np.random.uniform(-2,0,(dim_c,1)),
                          'b_z_'+name: np.random.uniform(-1,0,(dim_c,1)),
                          'b_c_'+name: np.random.uniform(-2,0,(dim_c,1))}
        
        rnn.InitialParameters = initial_params
        
    quality_model = QualityModel(subsystems=[one_model],
                                  name='q_model')
    
    
    s_opts = {"hessian_approximation": 'limited-memory',"max_iter": 3000,
              "print_level":2}
    
    
    results_GRU = ModelTraining(quality_model,data,initializations=20, BFR=False, 
                      p_opts=None, s_opts=s_opts)
    
    results_GRU['Chargen'] = 'c'+str(counter)
    
    pkl.dump(results_GRU,open('GRU_Durchmesser_innen_c'+str(counter)+'.pkl','wb'))
    
    print('Charge '+str(counter)+' finished.')
    
    return results_GRU  


results = Fit_GRU_to_Charges(list(range(1,275)),0)

