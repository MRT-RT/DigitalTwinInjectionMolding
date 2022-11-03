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

from DIM.models.models import Static_MLP
from DIM.optim.param_optim import ParamOptimizer
from DIM.arburg470 import dt_functions as dtf

# matplotlib.use("Qt4agg")

# %% Lese Trainingsdaten von Versuchsplan ein
# Nur für Offline-Demobetrieb

source_h5 = Path('C:/Users/alexa/Desktop/DIM/DIM_20221101.h5')
target_h5 = Path('C:/Users/alexa/Desktop/DIM/dm_data.h5')
source_live_h5 = Path('C:/Users/alexa/Desktop/DIM/DIM_20221102.h5')
model_path = Path('C:/Users/alexa/Desktop/DIM/live_models.pkl')

# source_h5 = Path('I:/Klute/DIM_Twin/DIM_20221101.h5')
# source_live_h5 = Path('I:/Klute/DIM_Twin/DIM_20221102.h5')
# target_h5 = Path('C:/Users/rehmer/Desktop/DIM_Data/dm_data.h5')
# model_path = Path('C:/Users/rehmer/Desktop/DIM_Data/models/live_models.pkl')

setpoints = ['v_inj_soll','V_um_soll','T_zyl5_soll']                           # T_wkz_soll fehlt

# Load DataManager specifically for this machine
dm = dtf.config_data_manager(source_h5,target_h5,setpoints)
# dm.get_cycle_data()


# %% Ändere Quelldatei für Live-Betrie
dm.source_hdf5 = source_live_h5
# %% 

mb = dtf.model_bank(model_path=model_path)

y_label = mb.models[0].y_label[0]

# %% Fonts for plots

font = {'weight' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)
     

# %% Main program
    
if __name__ == '__main__':
    
    # dm.get_cycle_data()
    
    freeze_support()

    plt.close('all')
    
    
    MQPlot = dtf.ModelQualityPlot()
    SQPlot = dtf.SolutionQualityPlot()
    PPlot = dtf.PredictionPlot() 
    OSPlot = dtf.OptimSetpointsPlot(num_sol=5)
    
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

        
        # Check for new data
        new_data = dm.get_cycle_data(20.0)
        
        # time.sleep(1)
        # Q_read = [None]
        # Read in new slider value
        # t = Thread(target=read_slider,args=(master,slider,Q_read) )
        # t.start()
        
        
        # Read target quality value from slider
        # master.lift()
        # time.sleep(2.0)
        master.update_idletasks()
        master.update()
        print(slider_val.get())
        new_val = slider.get()
        # print(new_val)
        
        new_val = 8.15

        new_data = True    

        if new_data:
            
            # Reload models
            mb.load_models()
                        
            # Predict new quality datum
            dtf.predict_quality(dm,mb)

            MQPlot.update(mb.stp_bfr[mb.idx_best])
            PPlot.update(dm,mb)
            plt.pause(0.01)
                        
            Q_target =  pd.DataFrame.from_dict({y_label: [new_val]})
            
            # calculate optimal setpoints
            opti_setpoints = dtf.optimize_setpoints(dm,mb,Q_target,10)
            
            # Plot 
            # SQPlot.update(opti_setpoints.loc[0,'loss'])
            OSPlot.update(opti_setpoints[dm.setpoints+['loss']])
            plt.pause(0.01)
            # master.lift()
        else:
            
            print('Waiting for new data')
            
            time.sleep(1)
            




    
#     p_read.join(0)
    
    # data_manager.get_cycle_data()

# 1. Read Data continuosly, give signal if new data available

# 2. Predict new quality datum by multiple models, return best prediction

# 3. Estimate optimal setpoint given best model, if model is even accurate 