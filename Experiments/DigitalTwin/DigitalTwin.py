# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 14:00:40 2022

@author: LocalAdmin
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

from DIM.models.model_structures import Static_MLP
from DIM.optim.param_optim import ParamOptimizer
from DIM.miscellaneous import DigitalTwinFunctions as dtf

# matplotlib.use("Qt4agg")

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


# %% Fonts for plots

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)


# %% Function for reading in slider value

def read_slider(window,slider,Q_read):
    
    old_val = Q_read[0]
    new_val = Q_read[0]
    
    while new_val == old_val:
        
        window.lift()
        
        # Read target quality value from slider
        window.update_idletasks()
        window.update()
        
        new_val = slider.get()
        
        time.sleep(1.0)
    
    Q_read[0]=new_val

        
    
    


# %% Main program
    
if __name__ == '__main__':
    
    freeze_support()
    
    l = 6 
    u = 100 #204
    plt.close('all')
    
    # Figure Setup
    fig1,ax1 = plt.subplots(1,2)
    
    mngr1 = plt.get_current_fig_manager()
    mngr1.window.setGeometry(0, 0, 3840 , 1000)
    
    fig2,ax2 = plt.subplots(1,len(mb.models[0].u_label))
    
    mngr2 = plt.get_current_fig_manager()
    mngr2.window.setGeometry(0, 1000, 3840 , 1000)
    
    # Slider Setup
    master = tk.Tk()
    slider = tk.Scale(master, from_=26, to=28,length=1000,width=100,
                  orient='vertical',digits=3,label='Durchmesser_innen',
                  resolution=0.1, tickinterval=0.5)
    slider.pack()
    
    
    master.attributes("-topmost", True)
    master.focus_force()
    
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
        
        # Q_read = [None]
        # # Read in new slider value
        # t = Thread(target=read_slider,args=(master,slider,Q_read) )
        # t.start()
        
        
        # Read target quality value from slider
        # master.lift()
        master.update_idletasks()
        master.update()
        new_val = slider.get()
        # new_val = 27.0
        
        if new_data:
                       
            # Predict new quality datum
            dtf.predict_quality(dm,mb)

            # plot measurement and prediction
            dtf.plot_meas_pred(fig1,ax1,dm,mb)
            
            # reestimate models
            # dtf.reestimate_models(dm,mb)
            
            # Reload models
            mb.load_models()
            
            Q_target =  pd.DataFrame.from_dict({'Durchmesser_innen': [new_val]})
            
            # calculate optimal setpoints
            opti_setpoints = dtf.optimize_setpoints(dm,mb,Q_target)
            

            
            
            opti_setpoints = pd.DataFrame(data=[[14.0,43,0],[14.5,43,1],
                                                [15.0,43,2]],
                                          columns=['V_um_soll','T_wkz_0',
                                                   'Sol_Num'])
            
            # plot optimal setpoints IMPLEMENT!!!
            dtf.parallel_plot_setpoints(fig2,ax2,opti_setpoints)
            
            
            # time.sleep(1)

    


#     p_read.join(0)
    
    # data_manager.get_cycle_data()

# 1. Read Data continuosly, give signal if new data available

# 2. Predict new quality datum by multiple models, return best prediction

# 3. Estimate optimal setpoint given best model, if model is even accurate 