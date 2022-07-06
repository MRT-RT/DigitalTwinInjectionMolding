# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:44:37 2021

@author: alexa
"""

import pickle as pkl
import numpy as np
import itertools

import multiprocessing

import sys
# sys.path.insert(0, "E:\GitHub\DigitalTwinInjectionMolding")
sys.path.insert(0, 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, '/home/alexander/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, 'E:/GitHub/DigitalTwinInjectionMolding/')


from DIM.miscellaneous.PreProcessing import arrange_data_for_ident, eliminate_outliers
from DIM.models.model_structures import TimeDelay_MLP
from DIM.models.injection_molding import QualityModel
from DIM.optim.param_optim import ParallelModelTraining, ModelTraining
from DIM.miscellaneous.PreProcessing import LoadDynamicData


def Fit_TDMLP(order,dim_h,initial_params=None):

    charges = list(range(1,26))
    
    split = 'all'
    mode='quality'
    
    path = 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/data/Stoergroessen/20220504/Versuchsplan/normalized/'
    # path = '/home/alexander/GitHub/DigitalTwinInjectionMolding/data/Stoergroessen/20220504/Versuchsplan/normalized/'
    
    u_inj= ['p_wkz_ist','T_wkz_ist']
    u_press= ['p_wkz_ist','T_wkz_ist']
    u_cool= ['p_wkz_ist','T_wkz_ist']
    
    u_lab = [u_inj,u_press,u_cool]
    y_lab = ['Gewicht']
    
    
    data_train,data_val = LoadDynamicData(path,charges,split,y_lab,u_lab,mode)
    
    c0_train = [np.zeros((order,1)) for i in range(0,len(data_train['data']))]
    c0_val = [np.zeros((order,1)) for i in range(0,len(data_val['data']))] 
    
    data_train['init_state'] = c0_train
    data_val['init_state'] = c0_val
    
    
    inj_model = TimeDelay_MLP(dim_u=2,dim_hidden=5,dim_out=1,dim_c=order,
                    u_label=u_inj,y_label=y_lab,name='inj')
    
    press_model = TimeDelay_MLP(dim_u=2,dim_hidden=5,dim_out=1,dim_c=order,
                    u_label=u_inj,y_label=y_lab,name='press')
    
    cool_model = TimeDelay_MLP(dim_u=2,dim_hidden=5,dim_out=1,dim_c=order,
                    u_label=u_inj,y_label=y_lab,name='cool')
       
    quality_model = QualityModel(subsystems=[inj_model,press_model,cool_model],
                                  name='q_model')
    
    s_opts = {"max_iter": 2000, 'hessian_approximation':'limited-memory'}

    
    results = ParallelModelTraining(quality_model,data_train,data_val,
                            initializations=10, BFR=False, p_opts=None, 
                            s_opts=s_opts,mode='parallel',n_pool=5)

    # results = ModelTraining(quality_model,data_train,data_val,
    #                         initializations=1, BFR=False, p_opts=None, 
    #                         s_opts=s_opts,mode='parallel')
        
    pkl.dump(results,open('TDMLP_order'+str(order)+'_hidden'+str(dim_h)+'_3sub_Stoergrsn_Gewicht.pkl','wb'))
  
    return results  


if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    orders = list(range(1,11))
    dim_h = [5,10,20,40]
    
    orders_dimh = list(itertools.product(orders, dim_h))
    
    for combo in orders_dimh:
        Fit_TDMLP(order=combo[0],dim_h=combo[1])