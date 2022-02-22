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
from DIM.optim.param_optim import ModelTraining
from DIM.miscellaneous.PreProcessing import LoadDynamicData


def Fit_GRU(dim_c,initial_params=None):

    charges = list(range(1,275))
    
    split = 'all'
    # split = 'part'
    
    # path = 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
    # path = '/home/alexander/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
    path = 'E:/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
       
   
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
    
    inj_model.InitialParameters = initial_params
    press_model.InitialParameters = initial_params #{'b_z_press':np.random.uniform(-100,-2,(dim_c,1))}
    cool_model.InitialParameters = initial_params #{'b_z_cool':np.random.uniform(-100,-2,(dim_c,1))}
   
    quality_model = QualityModel(subsystems=[inj_model,press_model,cool_model],
                                  name='q_model')
    
    # s_opts = {'max_iter': 2, 'step':0.1, 'hessian_approximation':'limited-memory'}
    s_opts = {'max_iter': 100, 'hessian_approximation':'limited-memory'}
    
    results_GRU = ModelTraining(quality_model,data,initializations=1, BFR=False, 
                      p_opts=None, s_opts=s_opts,mode='parallel')
        
    pkl.dump(results_GRU,open('GRU_c'+str(dim_c)+'_3sub_all.pkl','wb'))
  
    return results_GRU  

c2_part = pkl.load(open('GRU_c2_3sub.pkl','rb'))
c2_all = Fit_GRU(dim_c=2,initial_params=c2_part.loc[0]['params_val'])

c1_part = pkl.load(open('GRU_c1_3sub.pkl','rb'))
c1_all = Fit_GRU(dim_c=1,initial_params=c1_part.loc[4]['params_val'])

# if __name__ == '__main__':
#     multiprocessing.freeze_support()
#     c1 = Fit_GRU(dim_c=1)
#     c2 = Fit_GRU(dim_c=2)
#     c3 = Fit_GRU(dim_c=3)
#     c4 = Fit_GRU(dim_c=4)
