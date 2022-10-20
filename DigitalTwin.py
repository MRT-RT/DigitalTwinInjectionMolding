# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 14:00:40 2022

@author: LocalAdmin
"""
import multiprocessing
from multiprocessing import Process, freeze_support, Value
from pathlib import Path
import h5py

import DigitalTwinFunctions as dtf
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import numpy as np

from DIM.models.model_structures import Static_MLP
from DIM.optim.param_optim import Optimizer

# Load DataManager specifically for this machine
dm = dtf.config_data_manager()

# Nur für Offline-Demobetrieb
hist_path = Path('C:/Users/LocalAdmin/Documents/DIM_Data/Messung 5.10/hist_data.h5')
live_path = Path.cwd()/'live_data.h5'

model_paths = ['MLP'+str(i)+'.mod' for i in range(0,10)]

mb = dtf.model_bank(model_paths=model_paths)
       
if __name__ == '__main__':
    
    freeze_support()
    
    l = 6 
    u = 100 #204
    plt.close('all')
    fig,ax = plt.subplots(1,1)
    
    for i in range(l,u):
        print(i)
        # pretend new data is coming in
        hist = h5py.File(hist_path,mode='r')
        live = h5py.File(live_path,mode='a')
        try:
            hist.copy('cycle_'+str(i),live)
        except:
            pass
        hist.close()
        live.close()
        
        
        # Official code starts here
        # Check for new data
        new_data = dm.get_cycle_data()
    
        if new_data:
            
            # Reload models from file
            mb.load_models()
            
            # Predict new quality datum
            dtf.predict_quality(dm,mb)

            # find best model
            idx = np.argmin(mb.loss)            
            plt_data = pd.read_hdf(dm.target_hdf5,key='modelling_data')
            
            # prediction here
            # measured data
            #don't plot this data, plot the last n cycles or something
            sns.lineplot(data=plt_data,x = plt_data.index,
                         y = 'Durchmesser_innen',ax=ax, marker='o',color='k') 
            sns.lineplot(data=mb.pred[idx],x = mb.pred[idx].index,
                         y = 'Durchmesser_innen',ax=ax, marker='o',color='b')             
            
            plt.pause(0.1)
            
            
            # Start multiple processes to reestimate models
            ident_data = pd.read_hdf(dm.target_hdf5,key='modelling_data')
            
            for m in range(len(mb.models)):
                p = Process(target=dtf.reestimate_models,
                            args=(ident_data, mb.models[m],m))
                
                p.start()
                p.join()
            
            time.sleep(10.0)

    


#     p_read.join(0)
    
    # data_manager.get_cycle_data()

# 1. Read Data continuosly, give signal if new data available

# 2. Predict new quality datum by multiple models, return best prediction

# 3. Estimate optimal setpoint given best model, if model is even accurate 