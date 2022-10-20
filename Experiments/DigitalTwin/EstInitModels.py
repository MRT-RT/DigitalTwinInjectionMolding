# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 15:24:55 2022

@author: LocalAdmin
"""

import multiprocessing
from multiprocessing import Process, freeze_support, Value
from pathlib import Path
import h5py


import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import sys

path_dim = Path.cwd().parents[1]
sys.path.insert(0, path_dim.as_posix())

from DIM.miscellaneous import DigitalTwinFunctions as dtf
from DIM.models.model_structures import Static_MLP
from DIM.optim.param_optim import ModelTraining

# Load DataManager specifically for this machine
source_hdf = Path('C:/Users/LocalAdmin/Documents/DIM_Data/DOE_2/DOE_2_mod.h5')
target_hdf = Path.cwd()/'DOE_2_dm.h5'

dm = dtf.config_data_manager(source_hdf,target_hdf)
dm.get_cycle_data()

modelling_data = pd.read_hdf(dm.target_hdf5, 'scalar_data')

# for i in range(0,10):
#     MLP = Static_MLP(dim_u=2,dim_out=1,dim_hidden=5,u_label=['T_wkz_0','V_um_soll'],
#                      y_label=['Durchmesser_innen'],name='MLP'+str(i))
    
#     res = ModelTraining(MLP,modelling_data,modelling_data,mode='static',
#                         initializations=2)
    
#     idx = res['loss_val'].idxmin()
    
#     MLP.Parameters = res.loc[idx,'params_val']
    
#     pkl.dump(MLP,open('Models/'+MLP.name+'.mod','wb'))
    
    