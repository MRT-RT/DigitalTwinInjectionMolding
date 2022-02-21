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


from DIM.miscellaneous.PreProcessing import arrange_data_for_ident, eliminate_outliers
from DIM.models.model_structures import GRU
from DIM.models.injection_molding import QualityModel
from DIM.optim.param_optim import ParallelModelTraining
from DIM.miscellaneous.PreProcessing import LoadDynamicData


def Fit_GRU(dim_c,initial_params=None):

    charges = list(range(1,275))
    
    # split = 'all'
    split = 'part'
    
    path = 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
    # path = '/home/alexander/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
    # path = 'E:/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
    
    
    u_lab = [['p_wkz_ist','T_wkz_ist']]
    y_lab = ['Durchmesser_innen']
    
    data,cycles_train_label,cycles_val_label,charge_train_label,charge_val_label = \
    LoadDynamicData(path,charges,split,y_lab,u_lab)
    
    c0_train = [np.zeros((dim_c,1)) for i in range(0,len(data['u_train']))]
    c0_val = [np.zeros((dim_c,1)) for i in range(0,len(data['u_val']))]    
    
    data['init_state_train'] = c0_train
    data['init_state_val'] = c0_val
    
    
    one_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=1,dim_out=1,name='inj')

    one_model.InitialParameters = {'b_z_one':np.random.uniform(-100,-2,(dim_c,1))}
    
    quality_model = QualityModel(subsystems=[one_model],
                                  name='q_model')
    
    
    s_opts = {"max_iter": 200, 'step':0.1}
    
    results_GRU = ParallelModelTraining(quality_model,data,initializations=10, BFR=False, 
                      p_opts=None, s_opts=s_opts)
        
    pkl.dump(results_GRU,open('GRU_c'+str(dim_c)+'_1sub.pkl','wb'))
  
    return results_GRU  


if __name__ == '__main__':
    multiprocessing.freeze_support()
    c1 = Fit_GRU(dim_c=1)
    c2 = Fit_GRU(dim_c=2)
    c3 = Fit_GRU(dim_c=3)
    c4 = Fit_GRU(dim_c=4)