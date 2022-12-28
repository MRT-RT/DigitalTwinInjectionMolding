# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 14:00:40 2022

@author: LocalAdmin
"""
import multiprocessing
from multiprocessing import Process, freeze_support

# from threading import Thread
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
import ctypes

from DIM.models.models import Static_MLP
from DIM.optim.param_optim import ParamOptimizer
from DIM.arburg470 import dt_functions as dtf


# %% User specified parameters

# Pfad
path = Path('C:/Users/LocalAdmin/Desktop/DIM')

# Ggf. neue h5-Datei, in die opc_daq_main.py Daten schreibt
source_live_h5 = path/'DIM_20221108.h5'

# Pfad zu DataManager
dm_path = path/'dm_recyclat.pkl'

# Pfad zu live_models für die Modellprädiktion
model_path = path/'models/live_models.pkl'

# %% Lade Dictionaries die Modelle für jeden Zyklus sowie berechnete optimierte 
# Setpoints, falls nicht existent erzeuge neu 
try:
    mb_save = pkl.load(open(path/'mb_save.pkl','rb'))
    opt_save = pkl.load(open(path/'opt_save.pkl','rb'))
except:
    mb_save = {}
    opt_save = {}
    
# %%Lade DataManager
dm = pkl.load(open(dm_path,'rb'))
# %% Ändere Quelldatei für Live-Betrieb
dm.source_hdf5 = source_live_h5
# %% Load model bank
mb = dtf.model_bank(model_path=model_path)
y_label = mb.models[0].y_label[0]

# %% Initialize DataFrame for recording applied setpoints
rec = pd.DataFrame(data=None,columns=dm.setpoints+[y_label])


# %% Fonts for plots
font = {'weight' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)

# Get screen resolution    
user32 = ctypes.windll.user32
screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

# %% Main program
    
if __name__ == '__main__':
    
    # dm.get_cycle_data()
    
    freeze_support()

    plt.close('all')
    
    MQPlot = dtf.ModelQualityPlot(screensize[0],screensize[1])
    # SQPlot = dtf.SolutionQualityPlot(1080,400)
    PPlot = dtf.PredictionPlot(screensize[0],screensize[1])  
    OSPlot = dtf.OptimSetpointsPlot(screensize[0],screensize[1],num_sol=10)
    
    # Slider Setup
    master = tk.Tk()
    slider_val = tk.DoubleVar()
    slider = tk.Scale(master, from_=8.0, to=8.5,length=500,width=50,
                  orient='vertical',digits=3,label='Durchmesser_innen',
                  resolution=0.05, tickinterval=0.1,variable=slider_val)
    slider.pack()
    
    
    # master.attributes("-topmost", True)
    # master.focus_force()
    
    while True:
        # Save an updated version of the data manager object
        pkl.dump(dm,open(dm_path,'wb'))
        
        # Check for new data
        new_data = dm.get_cycle_data(delay=0.0,num_cyc=1)
        
        
        # Read target quality value from slider
        master.lift()
        master.update_idletasks()
        master.update()
        # print(slider_val.get())
        # new_val = slider.get()
        
        new_val = 8.15
        new_data = True
        
        if new_data:
            
            # Save applied setpoints and target quality value
            modelling_data = dm.get_static_modelling_data()
            idx_new = max(modelling_data.index)
            stp = modelling_data.loc[[idx_new],dm.setpoints]
            stp[y_label] = new_val
            
            rec = pd.concat([rec,stp])
            # pkl.dump(rec,open(rec_path,'wb'))
            
            # Reload models (in case Live_Model_Reestimation.py wrote new 
            # models to live_models.pkl)
            mb.load_models()
                       
            # Predict new quality datum
            dtf.predict_quality(dm,mb)
s
            # Speichere aktuelle Modelle und Modellprädiktion
            mb_save[idx_new] = mb
            pkl.dump(mb_save,open(path/'mb_save.pkl','wb'))
            
            
            MQPlot.update(mb.stp_bfr[mb.idx_best])
            PPlot.update(dm,y_label,mb=mb)
            master.lift()
            plt.pause(0.01)
                        
            Q_target =  pd.DataFrame.from_dict({y_label: [new_val]})
            
            # calculate optimal setpoints
            opti_setpoints = dtf.optimize_setpoints(dm,mb,Q_target,[])
            
            # Speichere berechnete optimale Setpoints
            opt_save[idx_new] = mb
            pkl.dump(opt_save,open(path/'opt_save.pkl','wb'))          
            
            # Plot 
            # SQPlot.update(opti_setpoints.loc[0,'loss'])
            if opti_setpoints is not None:
                OSPlot.update(opti_setpoints[dm.setpoints+['loss']],dm)
                
            plt.pause(0.01)
            master.lift()
        else:
            
            print('Waiting for new data')
            
            time.sleep(1)
            




    
#     p_read.join(0)
    
    # data_manager.get_cycle_data()

# 1. Read Data continuosly, give signal if new data available

# 2. Predict new quality datum by multiple models, return best prediction

# 3. Estimate optimal setpoint given best model, if model is even accurate 