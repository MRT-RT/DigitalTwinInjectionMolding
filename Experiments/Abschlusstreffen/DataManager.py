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

# source_hdf5 = Path('C:/Users/rehmer/Desktop/DIM/data/DIM_20221111_2.h5')
# target_hdf5 = Path('C:/Users/rehmer/Desktop/DIM/Optimierung/dm_Abschluss.h5')

source_hdf5 = Path('/home/alexander/Desktop/DIM/DIM_20221104.h5')
target_hdf5 = Path('/home/alexander/Desktop/DIM/dm_test.h5')

setpoints = ['v_inj_soll','V_um_soll','T_wkz_soll']   

# Generate DataManager object
dm = dtf.config_data_manager(source_hdf5,target_hdf5,setpoints)
dm.get_cycle_data()


# %% Load data as example
# machine_data = dm.get_machine_data()
m_data = dm.get_modelling_data()
# process_values_67820 = dm.get_process_data(modelling_data.index[0])

fig,ax = plt.subplots(1,1)
sns.stripplot(data=m_data,x=m_data.index,y='Gewicht') 

# Save Data Manager object as a pickle
# pkl.dump(dm,open('C:/Users/rehmer/Desktop/DIM/Optimierung/dm.pkl','wb'))
pkl.dump(dm,open('/home/alexander/Desktop/DIM/dm.pkl','wb'))
