# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 15:16:22 2022

@author: LocalAdmin
"""

import sys
sys.path.insert(0, '/home/alexander/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, 'E:/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/')

# import os.path as path
# two_up =  path.abspath(path.join(__file__ ,"../.."))
# print(two_up)

from DIM.miscellaneous.PreProcessing import LoadFeatureData, MinMaxScale
from DIM.models.model_structures import Static_MLP
from DIM.optim.param_optim import ParallelModelTraining

import multiprocessing

import pickle as pkl


def Fit_MLP(dim_hidden):
    
    print(dim_hidden)
    charges = list(range(1,275))
    
    split = 'all'
    
    path_sys = 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/'
    # path_sys = '/home/alexander/GitHub/DigitalTwinInjectionMolding/' 
    # path_sys = 'E:/GitHub/DigitalTwinInjectionMolding/'
    
    path = path_sys + '/data/Versuchsplan/normalized/'
    
    data_train,data_val = LoadFeatureData(path,charges,split)
    
    u_label = ['Düsentemperatur', 'Werkzeugtemperatur',
                'Einspritzgeschwindigkeit', 'Umschaltpunkt', 'Nachdruckhöhe',
                'Nachdruckzeit', 'Staudruck', 'Kühlzeit','T_wkz_0','p_inj_0',
                'x_0']
    
    y_label = ['Gewicht']   
    
    # Normalize Data
    data_train,minmax = MinMaxScale(data_train,u_label+y_label)
    data_val,_ = MinMaxScale(data_val,u_label+y_label,minmax)
    
    model = Static_MLP(dim_u=11, dim_out=1, dim_hidden=dim_hidden,u_label=u_label,
                        y_label=y_label,name='MLP', init_proc='xavier')
    
    s_opts = {"max_iter": 2000, 'hessian_approximation':'limited-memory'}
    
    result = ParallelModelTraining(model,data_train,data_val,initializations=10,
                           p_opts=None,s_opts=s_opts,mode='static',n_pool=5)

    result['dim_hidden'] = dim_hidden
    
    pkl.dump(result,open('QualityModel_Gewicht_static_MLP_'+str(dim_hidden)+'.pkl','wb'))

    return result

    
if __name__ == '__main__':
    
    h1 = Fit_MLP(dim_hidden=1)
    h2 = Fit_MLP(dim_hidden=2)
    h3 = Fit_MLP(dim_hidden=3)
    h4 = Fit_MLP(dim_hidden=4)
    h5 = Fit_MLP(dim_hidden=5)   
    h6 = Fit_MLP(dim_hidden=6)     
    h7 = Fit_MLP(dim_hidden=7)
    h8 = Fit_MLP(dim_hidden=8)
    h9 = Fit_MLP(dim_hidden=9)
    h10 = Fit_MLP(dim_hidden=10)
    h20 = Fit_MLP(dim_hidden=20)
    h40 = Fit_MLP(dim_hidden=40)    










