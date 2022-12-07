#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 13:14:14 2022

@author: alexander
"""

import multiprocessing
from multiprocessing import Process, freeze_support

from threading import Thread
from pathlib import Path
import sys
import h5py
import tkinter as tk

path_dim = Path.cwd().parents[1]
sys.path.insert(0,path_dim.as_posix())


# import DigitalTwinFunctions as dtf
import time
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
import numpy as np

from DIM.models.models import Static_Multi_MLP
from DIM.optim.param_optim import ParamOptimizer
from DIM.arburg470 import dt_functions as dtf

# %%
# model_path = Path('C:/Users/rehmer/Desktop/DIM/models')
model_path = Path('/home/alexander/Desktop/DIM/models')




# %% Lade Data Manager
# dm = pkl.load(open('C:/Users/rehmer/Desktop/DIM/Optimierung/dm.pkl','rb'))
dm = pkl.load(open('/home/alexander/Desktop/DIM/dm.pkl','rb'))

if __name__ == '__main__':
    
    freeze_support()  
    inits = 20
    
    l = 1
    h = 6
    
    target = ['Gewicht']
    name = 'm_MLP_l'+str(l)+'_h'+str(h)
    
    go = True
    
    while go:
        
        modelling_data = dm.get_modelling_data()
        
        MLP = Static_Multi_MLP(dim_u=3,dim_out=1,dim_hidden=h,layers = l,
                               u_label=dm.setpoints,
                               y_label=target,name=name)
        
        # Drop NaN
        modelling_data = modelling_data[MLP.u_label+MLP.y_label].dropna()
        
        # Normalize data
        data_norm = MLP.MinMaxScale(modelling_data)
        
        # Initialize optimizer
        opt = ParamOptimizer(MLP,data_norm,data_norm,mode='static',
                            initializations=inits,
                            res_path=model_path,
                            n_pool=20)
        
        # Start optimizer
        results = opt.optimize()

        
        # Finde die 10 besten Modelle
        results_sort = results.sort_values(by='loss_val',ascending=True)
        
        # Lade alle Modelle
        models = pkl.load(open(model_path/'models.pkl','rb'))

        # Behalte nur 10 beste Modelle
        models_best = {list(models.keys())[i]: models[i] for i in results_sort.index[0:1]}
        
        pkl.dump(models_best,open(model_path/'live_models.pkl','wb'))
               
        go = False
        
        print('Estimation complete')
        
        