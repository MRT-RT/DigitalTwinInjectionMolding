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
dm.source_hdf5 = Path('C:/Users/alexa/Desktop/DIM/data/DIM_20221108.h5')




# %% Load model bank
model_path = Path('C:/Users/alexa/Desktop/DIM/models/live_models.pkl')
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
    
    PPlot = dtf.PredictionPlot() 
    
    while True:
       
        # Check for new data
        new_data = dm.get_cycle_data(delay=2.0,num_cyc=1)
      
        if new_data:
            
            # Reload models
            mb.load_models()
            
            dtf.predict_quality(dm,mb)          
            PPlot.update(dm,'Gewicht',mb=mb)
            
            # Pausiere Plot damit dieser aktualisiert wird
            plt.pause(0.01)
            
        else:
            
            print('Waiting for new data')
            time.sleep(1)