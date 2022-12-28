#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 15:57:38 2022

@author: alexander
"""

# %%

from pathlib import Path
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl

path_dim = Path.cwd().parents[1]
sys.path.insert(0,path_dim.as_posix())

from DIM.arburg470 import dt_functions as dtf


# %% Define source file and file to write data to


path = Path('C:/Users/LocalAdmin/Desktop/DIM')

# source_hdf5 = Path('C:/Users/rehmer/Desktop/DIM/data/DIM_20221111_2.h5')
# target_hdf5 = Path('C:/Users/rehmer/Desktop/DIM/Optimierung/dm_Abschluss.h5')
# h5-Datei, in die opc_daq_main.py Daten schreibt
source_hdf5 = path/'DIM_20221104.h5'

# h5-Datei, in die die konvertierten Daten geschrieben werden sollen, kann 
# bereits existieren oder nicht
target_hdf5 = path/'target_h5_recyclat.h5'

# Die Setpoints, die für den vorliegenden Anwendungsfall an der Maschine mani-
# puliert werden. Alle anderen Maschinenparameter werden dann als konstant an-
# genommen
setpoints = ['v_inj_soll','V_um_soll','T_wkz_soll']   

# %%
# config_data_manager konfiguriert den DataManager so, dass er weiß welche 
# Variablen er auszulesen hat, welche Prozessgrößen in welchen Charts organi-
# siert sind uws.
dm = dtf.config_data_manager(source_hdf5,target_hdf5,setpoints)

# get_cycle_data konvertiert alle Zyklen aus source_hdf5 in target_hdf5, die 
# nicht bereits in target_hdf5 vorhanden sind 
dm.get_cycle_data()

# Speichere den DataManager
pkl.dump(dm,open(path/'dm_recyclat.pkl','wb'))


# %% Load data as example
# machine_data = dm.get_machine_data()
# m_data = dm.get_modelling_data()
# process_values_67820 = dm.get_process_data(modelling_data.index[0])

# fig,ax = plt.subplots(1,1)
# sns.stripplot(data=m_data,x=m_data.index,y='Gewicht') 


