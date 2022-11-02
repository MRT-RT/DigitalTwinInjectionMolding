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

# matplotlib.use("Qt4agg")

# %% Lese Trainingsdaten von Versuchsplan ein
# Nur für Offline-Demobetrieb

source_h5 = Path('/home/alexander/Desktop/DIM/DIM_20221101.h5')
target_h5 = Path('/home/alexander/Desktop/DIM/01_11_test.h5')

model_path = Path('/home/alexander/Desktop/DIM/')

setpoints = ['v_inj_soll','V_um_soll','T_zyl5_soll']     

# Load DataManager specifically for this machine
dm = dtf.config_data_manager(source_h5,target_h5,setpoints)

# 

if __name__ == '__main__':
    
    freeze_support()  
    inits = 20
    
    l = 2
    h=10
    
    target = ['Gewicht']
    
    name = 'Di_MLP_l'+str(l)+'_h'+str(h)
    
    go = True
    
    while go:
        
        modelling_data = pd.read_hdf(dm.target_hdf5, 'modelling_data')
        
    
        MLP = Static_Multi_MLP(dim_u=4,dim_out=1,dim_hidden=h,layers = l,
                               u_label=setpoints + ['T_wkz_0'],
                               y_label=target,name=name)
        
        data_norm = MLP.MinMaxScale(modelling_data)
        
        opt = ParamOptimizer(MLP,data_norm,data_norm,mode='static',
                            initializations=inits,
                            res_path=model_path/name,
                            n_pool=20)
    
        results = opt.optimize()
        
        go = False
        
        # Finde die 10 besten Modelle
        results_sort = results.sort_values(by='loss_val',ascending=True)
        
        # Lade alle Modelle
        models = pkl.load(open(model_path/name/'models.pkl','rb'))

        # Behalte nur 10 beste Modelle
        models_best = {list(models.keys())[i]: models[i] for i in results_sort.index[0:10]}
        
        pkl.dump(models_best,open(model_path/name/'live_models.pkl','wb'))
        
        
        
        
        