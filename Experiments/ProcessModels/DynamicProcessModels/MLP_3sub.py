#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 10:17:29 2022

@author: alexander
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

import multiprocessing

import sys
# sys.path.insert(0, "E:\GitHub\DigitalTwinInjectionMolding")
sys.path.insert(0, 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, '/home/alexander/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, 'E:/GitHub/DigitalTwinInjectionMolding/')

from DIM.miscellaneous.PreProcessing import LoadDynamicData
from DIM.models.model_structures import MLP
from DIM.models.injection_molding import ProcessModel
from DIM.optim.param_optim import parallel_mode
from DIM.optim.param_optim import ParallelModelTraining



def Fit_MLP(dim_hidden,initial_params=None):

    charges = list(range(1,275))    
 
    split = 'process'
    mode = 'process'

    path = '/home/alexander/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
    # path = 'E:/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'

    u_inj= ['v_inj_soll']
    u_press= ['p_inj_soll']
    u_cool= []

    u_lab = [u_inj,u_press,u_cool]
    y_lab = ['Q_Vol_ist','V_Screw_ist','p_wkz_ist','T_wkz_ist','p_inj_ist']

    data,cycles_train_label,cycles_val_label,charge_train_label,charge_val_label = \
    LoadDynamicData(path,charges,split,y_lab,u_lab,mode)

    inj_model = MLP(dim_u=1,dim_out=5,dim_hidden=dim_hidden,name='inj')
    press_model = MLP(dim_u=1,dim_out=5,dim_hidden=dim_hidden,name='press')
    cool_model = MLP(dim_u=0,dim_out=5,dim_hidden=dim_hidden,name='cool')

    process_model = ProcessModel(subsystems=[inj_model,press_model,cool_model],
                                  name='p_model')   


    s_opts = {"max_iter": 2000, 'hessian_approximation':'limited-memory'}

    results_MLP =  ParallelModelTraining(process_model,data,initializations=20, 
                    BFR=False, p_opts=None, s_opts=s_opts,mode='parallel',n_pool=1)
    
    pkl.dump(results_MLP,open('MLP_h'+str(dim_hidden)+'_3sub.pkl','wb'))


    return results_MLP



if __name__ == '__main__':
    multiprocessing.freeze_support()
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

