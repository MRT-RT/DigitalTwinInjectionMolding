# -*- coding: utf-8 -*-
# %% 
"""
Created on Mon Oct 10 14:00:40 2022

@author: LocalAdmin
"""
from multiprocessing import freeze_support

# from threading import Thread
from pathlib import Path
import sys
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
dm = pkl.load(open('C:/Users/rehmer/Desktop/DIM/Datenaufzeichnung/dm.pkl','rb'))

# %% Ändere Quelldatei für Live-Betrieb
dm.source_hdf5 = Path('I:/Klute/DIM_Twin/DIM_20221130.h5')

# %% Fonts for plots
font = {'weight' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)

width = 1920
height = 1080
     
# %% Main program
    
if __name__ == '__main__':
    
    # dm.get_cycle_data()
    
    freeze_support()

    plt.close('all')
    
    PPlot = dtf.PredictionPlot(width,height) 
    
    while True:
       
        # Check for new data
        new_data = dm.get_cycle_data(delay=20.0,num_cyc=1)
      
        if new_data:
                      
            PPlot.update(dm,'Gewicht')
            
            # Pausiere Plot damit dieser aktualisiert wird
            plt.pause(0.01)
            
        else:
            
            print('Waiting for new data')
            time.sleep(1)
            
