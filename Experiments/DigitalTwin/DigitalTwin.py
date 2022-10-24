# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 14:00:40 2022

@author: LocalAdmin
"""
import multiprocessing
from multiprocessing import Process, freeze_support, Value
from pathlib import Path
import sys
import h5py

path_dim = Path.cwd().parents[1]
sys.path.insert(0,path_dim.as_posix())


# import DigitalTwinFunctions as dtf
import time
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
import numpy as np

from DIM.models.model_structures import Static_MLP
from DIM.optim.param_optim import ParamOptimizer
from DIM.miscellaneous import DigitalTwinFunctions as dtf

# %% 

# Nur für Offline-Demobetrieb
hist_path = Path('C:/Users/LocalAdmin/Documents/DIM_Data/Messung 5.10/hist_data.h5')
live_path = Path.cwd()/'live_data.h5'

# Load DataManager specifically for this machine
dm = dtf.config_data_manager(live_path,Path('test.h5'),
                             ['v_inj_soll','V_um_soll'])
# dm = dtf.config_data_manager(hist_path,Path('all_data_05_10_22.h5'))

# Load a model bank
model_paths = ['Models/MLP'+str(i)+'.mod' for i in range(0,3)]
mb = dtf.model_bank(model_paths=model_paths)


# %%

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)

# %%
    
if __name__ == '__main__':
    
    freeze_support()
    
    l = 6 
    u = 100 #204
    plt.close('all')
    
    # Figure Setup
    fig1,ax1 = plt.subplots(1,2)
    
    mngr1 = plt.get_current_fig_manager()
    mngr1.window.setGeometry(0, 0, 3840 , 1000)
    
    fig2,ax2 = plt.subplots(1,1)
    mngr2 = plt.get_current_fig_manager()
    mngr2.window.setGeometry(0, 1000, 3840 , 1000)    
    
    
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
                       
            # Predict new quality datum
            dtf.predict_quality(dm,mb)

            # plot measurement and prediction
            dtf.plot_meas_pred(fig1,ax1,dm,mb)
            
            # reestimate models
            # dtf.reestimate_models(dm,mb)
            
            # Reload models
            mb.load_models()
            
            # calculate optimal setpoints
            opti_setpoints = dtf.optimize_setpoints(dm,mb)
            
            # plot optimal setpoints IMPLEMENT!!!
            dtf.plot_opti_setpoints(opti_setpoints)
            time.sleep(1)

    


#     p_read.join(0)
    
    # data_manager.get_cycle_data()

# 1. Read Data continuosly, give signal if new data available

# 2. Predict new quality datum by multiple models, return best prediction

# 3. Estimate optimal setpoint given best model, if model is even accurate 