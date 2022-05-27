# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:44:37 2021

@author: alexa
"""

import pickle as pkl
import numpy as np

# import multiprocessing

import sys

sys.path.insert(0, 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, '/home/alexander/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, 'E:/GitHub/DigitalTwinInjectionMolding/')


# from DIM.miscellaneous.PreProcessing import arrange_data_for_ident, eliminate_outliers
from DIM.models.model_structures import LSS
from DIM.models.injection_molding import QualityModel
from DIM.optim.param_optim import ParallelModelTraining
from DIM.miscellaneous.PreProcessing import LoadDynamicData


def Fit_LSS(dim_c,initial_params=None):

    charges = list(range(1,2))
    
    # split = 'all'
    split = 'part'
    
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
    
    
    inj_model = LSS(dim_u=2,dim_c=dim_c,dim_hidden=1,dim_out=1,name='inj')
    press_model = LSS(dim_u=2,dim_c=dim_c,dim_hidden=1,dim_out=1,name='press')
    cool_model = LSS(dim_u=2,dim_c=dim_c,dim_hidden=10,dim_out=1,name='cool')
    
    inj_model.A_eig = np.random.uniform(0.995,1,(dim_c))
    press_model.A_eig = np.random.uniform(0.995,1,(dim_c))
    cool_model.A_eig = np.random.uniform(0.995,1,(dim_c))
        
    quality_model = QualityModel(subsystems=[inj_model,press_model,cool_model],
                                  name='q_model')
    
    
    s_opts = {"max_iter": 100, "step":1}

    
    results = ParallelModelTraining(quality_model,data,initializations=2, BFR=False, 
                      p_opts=None, s_opts=s_opts)
    
    # results_GRU['Chargen'] = 'c'+str(counter)
    
    pkl.dump(results,open('LSS_c'+str(dim_c)+'_3sub_results.pkl','wb'))
    
    
    return results  

# c1 = Fit_LSS(1)
c2 = Fit_LSS(2)
# c3 = Fit_LSS(3)
# c4 = Fit_LSS(4)

# if __name__ == '__main__':
#     multiprocessing.freeze_support()
#     GRU_init = Fit_LSS()
 
