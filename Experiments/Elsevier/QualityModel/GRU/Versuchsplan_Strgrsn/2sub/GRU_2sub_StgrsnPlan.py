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
from DIM.optim.param_optim import ParallelModelTraining, ModelTraining
from DIM.miscellaneous.PreProcessing import LoadDynamicData


def Fit_GRU(dim_c,initial_params=None):

    charges = list(range(1,26))
    
    split = 'all'
    mode='quality'
    
    # path = 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/data/Stoergroessen/20220504/Versuchsplan/normalized/'
    path = '/home/alexander/GitHub/DigitalTwinInjectionMolding/data/Stoergroessen/20220504/Versuchsplan/normalized/'
    
    u_inj= ['p_wkz_ist','T_wkz_ist']
    u_press= ['p_wkz_ist','T_wkz_ist']
    # u_cool= ['p_wkz_ist','T_wkz_ist']
    
    u_lab = [u_inj,u_press]
    y_lab = ['Gewicht']
    
    
    data_train,data_val = \
    LoadDynamicData(path,charges,split,y_lab,u_lab,mode,None)
    
    c0_train = [np.zeros((dim_c,1)) for i in range(0,len(data_train['data']))]
    c0_val = [np.zeros((dim_c,1)) for i in range(0,len(data_val['data']))] 
    
    data_train['init_state'] = c0_train
    data_val['init_state'] = c0_val
    
    
    inj_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=1,
                    u_label=u_inj,y_label=y_lab,dim_out=1,name='inj')
    
    press_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=10,
                      u_label=u_press,y_label=y_lab,dim_out=1,name='press')
    
    # cool_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=10,
    #                  u_label=u_cool,y_label=y_lab,dim_out=1,name='cool')
    
    inj_model.InitialParameters = initial_params
    press_model.InitialParameters = initial_params 
    # cool_model.InitialParameters = initial_params 
    
    press_model.InitialParameters ={'b_z_press':np.random.uniform(-10,-2,(dim_c,1))}
    # cool_model.InitialParameters = {'b_z_cool':np.random.uniform(-10,-4,(dim_c,1))}
    
    quality_model = QualityModel(subsystems=[inj_model,press_model],
                                  name='q_model')
    
    s_opts = None #{"max_iter": 2000, 'hessian_approximation':'limited-memory'}

    
    # results_GRU = ParallelModelTraining(quality_model,data_train,data_val,
    #                         initializations=10, BFR=False, p_opts=None, 
    #                         s_opts=s_opts,mode='parallel',n_pool=10)

    results_GRU = ModelTraining(quality_model,data_train,data_val,
                            initializations=10, BFR=False, p_opts=None, 
                            s_opts=s_opts,mode='parallel')
        
    pkl.dump(results_GRU,open('GRU_c'+str(dim_c)+'_2sub_Stoergrsn_Gewicht.pkl','wb'))
  
    return results_GRU  

# c1_part = pkl.load(open('GRU_c1_3sub.pkl','rb'))
# c1_all = Fit_GRU(dim_c=1,initial_params=c1_part.loc[4]['params_val'])

# c2_part = pkl.load(open('GRU_c2_3sub.pkl','rb'))
# c2_all = Fit_GRU(dim_c=2,initial_params=c2_part.loc[0]['params_val'])

# c3_part = pkl.load(open('GRU_c3_3sub.pkl','rb'))
# c3_all = Fit_GRU(dim_c=3,initial_params=c3_part.loc[0]['params_val'])

# c4_part = pkl.load(open('GRU_c4_3sub.pkl','rb'))
# c4_all = Fit_GRU(dim_c=4,initial_params=c4_part.loc[4]['params_val'])

if __name__ == '__main__':
    multiprocessing.freeze_support()
    c1 = Fit_GRU(dim_c=1)
    c2 = Fit_GRU(dim_c=2)
    c3 = Fit_GRU(dim_c=3)
    c4 = Fit_GRU(dim_c=4)
    c5 = Fit_GRU(dim_c=5)
    c6 = Fit_GRU(dim_c=6)
    c7 = Fit_GRU(dim_c=7)
    c8 = Fit_GRU(dim_c=8)
    c9 = Fit_GRU(dim_c=9)
    c10 = Fit_GRU(dim_c=10)
    
