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
from DIM.models.model_structures import Static_Multi_MLP
from DIM.optim.param_optim import ModelTraining,Optimizer

import multiprocessing

import pickle as pkl


def Fit_MLP(dim_hidden,data_train,data_val):
    
    
    u_label = ['Düsentemperatur', 'Werkzeugtemperatur',
                'Einspritzgeschwindigkeit', 'Umschaltpunkt', 'Nachdruckhöhe',
                'Nachdruckzeit', 'Staudruck', 'Kühlzeit']
    
    y_label = ['Durchmesser_innen']   
    
    # Normalize Data
    data_train,minmax = MinMaxScale(data_train,columns=u_label+y_label)
    data_val,_ = MinMaxScale(data_val,minmax=minmax)
    
    model = Static_Multi_MLP(dim_u=8, dim_out=1, dim_hidden=dim_hidden,
                             layers=2,u_label=u_label,y_label=y_label,
                             name='MLP', init_proc='xavier')
    
    s_opts = {"max_iter": 2000, 'hessian_approximation':'limited-memory'}
    
    optimizer = Optimizer(model,data_train,data_val,initializations=20,
                           mode='static')
    
    results = optimizer.optimize()
    # result = ModelTraining(model,data_train,data_val,initializations=20,
    #                        p_opts=None,s_opts=s_opts,mode='static')

    

    # result['dim_hidden'] = dim_hidden

    return None#{'model':model,'est_params':result,'minmax':minmax}

data = pkl.load(open('data_doubleExp.pkl','rb'))
data_train = data['data_train']
data_test = data['data_test']

    
if __name__ == '__main__':
    
    multiprocessing.freeze_support()
    
    for i in range(10,11):
        res = Fit_MLP(i,data_train,data_test) 
        
        # pkl.dump(res,open('MLP_2layers_static/QM_MLP_Di_h'+str(i)+'.pkl','wb'))
   










