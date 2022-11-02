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

# source_h5 = Path('I:/Klute/DIM_Twin/DIM_20221102.h5')
# target_h5 = Path('C:/Users/rehmer/Desktop/DIM_Data/01_11_test.h5')

source_h5 = Path('/home/alexander/Desktop/DIM/DIM_20221101.h5')
target_h5 = Path('/home/alexander/Desktop/DIM/01_11_test.h5')

setpoints = ['v_inj_soll','V_um_soll','T_zyl5_soll']                           # T_wkz_soll fehlt

# Load DataManager specifically for this machine
dm = dtf.config_data_manager(source_h5,target_h5,setpoints)
# dm.get_cycle_data()

# dm = dtf.config_data_manager(hist_path,Path('all_data_05_10_22.h5'))

# %% Ändere Quelldatei für Live-Betrie
source_live_h5 = Path('/home/alexander/Desktop/DIM/DIM_20221102.h5')
dm.source_hdf5 = source_live_h5
# %% 

model_path = Path('/home/alexander/Desktop/DIM/Di_MLP_l2_h10/live_models.pkl')

mb = dtf.model_bank(model_path=model_path)

y_label = mb.models[0].y_label[0]

# %% Fonts for plots

font = {'weight' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)
     

# %% Main program
    
if __name__ == '__main__':
    
    dm.get_cycle_data()
    
    freeze_support()
    
    # l = 6 
    # u = 100 #204
    plt.close('all')
    
    
    MQPlot = dtf.ModelQualityPlot()
    SQPlot = dtf.SolutionQualityPlot()
    PPlot = dtf.PredictionPlot() 
    OSPlot = dtf.OptimSetpointsPlot()
    
    # Slider Setup
    master = tk.Tk()
    slider_val = tk.DoubleVar()
    slider = tk.Scale(master, from_=27, to=29,length=500,width=50,
                  orient='vertical',digits=3,label='Durchmesser_innen',
                  resolution=0.1, tickinterval=0.5,variable=slider_val)
    slider.pack()
    
    
    # master.attributes("-topmost", True)
    # master.focus_force()
    
    while True:

        
        # Check for new data
        new_data = dm.get_cycle_data(16.0)
        
        # time.sleep(1)
        # Q_read = [None]
        # Read in new slider value
        # t = Thread(target=read_slider,args=(master,slider,Q_read) )
        # t.start()
        
        
        # Read target quality value from slider
        # master.lift()
        # time.sleep(2.0)
        # master.update_idletasks()
        # master.update()
        # print(slider_val.get())
        # new_val = slider.get()
        # print(new_val)
        new_val = 8.15
        
        # new_data = True
        
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
            opti_setpoints = dtf.optimize_setpoints(dm,mb,Q_target,1)
            
            # Plot 
            SQPlot.update(opti_setpoints.loc[0,'loss'])
            OSPlot.update(opti_setpoints[dm.setpoints])
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