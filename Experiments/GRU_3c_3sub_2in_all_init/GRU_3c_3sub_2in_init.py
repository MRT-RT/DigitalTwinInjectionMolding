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


def Fit_GRU(initial_params=None):

    charges = list(range(1,275))
    dim_c = 1
    
    # split = 'all'
    split = 'part'
    
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
    
    press_model.InitialParameters = {'b_z_press':np.ones((dim_c,1))*-(10)}
    cool_model.InitialParameters = {'b_z_cool':np.ones((dim_c,1))*-(10)}

    # press_model.InitialParameters = {'b_z_press':np.random.uniform(-10,-1,(dim_c,1))}
    # cool_model.InitialParameters = {'b_z_cool':np.random.uniform(-10,-1,(dim_c,1))}     
        
    quality_model = QualityModel(subsystems=[inj_model,press_model,cool_model],
                                  name='q_model')
    
    
    s_opts = {"max_iter": 500, "step":0.1}

    
    results_GRU = ParallelModelTraining(quality_model,data,initializations=20, BFR=False, 
                      p_opts=None, s_opts=s_opts)
    
    # results_GRU['Chargen'] = 'c'+str(counter)
    
    pkl.dump(results_GRU,open('GRU_c'+str(dim_c)+'_results.pkl','wb'))
    
    print('Charge '+str(counter)+' finished.')
    
    return results_GRU  



if __name__ == '__main__':
    multiprocessing.freeze_support()
    GRU_init = Fit_GRU()
 

# res = pkl.load(open('GRU_Durchmesser_innen_c2.pkl','rb'))
# res_sorted = res.sort_values('loss_val')

# initial_params = [res_sorted.iloc[0]['params'],res_sorted.iloc[1]['params'],
#                   res_sorted.iloc[2]['params']]
# counter = [2,9,0]

# for i in range(0,3):
    
# Fit_GRU(counter[i],initial_params[i])
    
  
# if __name__ == '__main__':
    
#     print('Process started..')
    
#     res = pkl.load(open('GRU_Durchmesser_innen_c2.pkl','rb'))
#     res_sorted = res.sort_values('loss_val')

#     initial_params = [res_sorted.iloc[0]['params'],res_sorted.iloc[1]['params'],
#                       res_sorted.iloc[2]['params']]

#     initial_params = [res_sorted.iloc[0]['params']]    
    
#     multiprocessing.freeze_support()
    
#     pool = multiprocessing.Pool()
    
#     result = pool.starmap(Fit_GRU, zip([2] ,initial_params)) 

