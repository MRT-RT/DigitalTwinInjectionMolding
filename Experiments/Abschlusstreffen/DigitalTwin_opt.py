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

from DIM.models.models import Static_MLP
from DIM.optim.param_optim import ParamOptimizer
from DIM.arburg470 import dt_functions as dtf


# %% Lade Data Manager
dm = pkl.load(open('dm.pkl','rb'))

# %% Ändere Quelldatei für Live-Betrieb
# dm.source_hdf5 = Path('C:/Users/alexa/Desktop/DIM/data/DIM_20221108.h5')
dm.source_hdf5 = Path('I:/Klute/DIM_Twin/DIM_20221125.h5')

# %% Load model bank
model_path = Path('C:/Users/rehmer/Desktop/DIM/models/live_models.pkl')

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
    
    # Initialisiere Plots
    PPlot = dtf.PredictionPlot() 
    MQPlot = dtf.ModelQualityPlot()
    OSPlot = dtf.OptimSetpointsPlot(num_sol=10) 
    
    
    # Slider Setup
    master = tk.Tk()
    slider_val = tk.DoubleVar()
    slider = tk.Scale(master, from_=8.1, to=8.3,length=500,width=50,
                  orient='vertical',digits=3,label='Durchmesser_innen',
                  resolution=0.025, tickinterval=0.1,variable=slider_val)
    slider.pack()
    
    master.attributes("-topmost", True)
    master.focus_force()
    
    # master.mainloop()
    
    while True:
        
        # Read target quality value from slider
        master.lift()
        for i in range(10):
            plt.pause(0.2)
            master.update_idletasks()
            master.update()
        print(slider_val.get())
        # new_val = slider.get()
        
        new_val = 8.15
        new_data = True
        
        # Check for new data
        new_data = dm.get_cycle_data(delay=20.0,num_cyc=1)
      
        if new_data:
            
            # Reload models
            mb.load_models()
            
            # Führe Prädiktion durch
            dtf.predict_quality(dm,mb)     
            
            # Plotte BFR des besten Modells
            MQPlot.update(mb.stp_bfr[mb.idx_best])
            
            # Plotte Prädiktion des besten Modells
            PPlot.update(dm,'Gewicht',mb=mb)
            
            # Pausiere Plot damit dieser aktualisiert wird
            plt.pause(0.01)
            
            # Berechne optimale Maschinenparameter
            Q_target =  pd.DataFrame.from_dict({y_label: [new_val]})
            opti_setpoints = dtf.optimize_setpoints(dm,mb,Q_target,[])
            
            # Plotte optimale Maschinenparameter
            if opti_setpoints is not None:            
                OSPlot.update(opti_setpoints,dm)
                
            plt.pause(0.01)
            master.lift()                
            
        else:
            
            print('Waiting for new data')
            time.sleep(1)