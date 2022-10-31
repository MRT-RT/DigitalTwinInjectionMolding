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

from DIM.arburg470 import dt_functions as dtf
from DIM.models.models import Static_MLP
from DIM.optim.param_optim import ParamOptimizer

# Load DataManager specifically for this machine
source_h5 = Path('I:/Klute/DIM_Twin/DIM_20221029.h5')
target_h5 = Path('C:/Users/rehmer/Desktop/DIM_Data/31_10_setup.h5')

setpoints = ['v_inj_soll','V_um_soll','p_pack2_soll']   


# Load DataManager specifically for this machine
dm = dtf.config_data_manager(source_h5,target_h5,setpoints)

# Get data for model parameter estimation
dm.get_cycle_data()
modelling_data = pd.read_hdf(dm.target_hdf5, 'modelling_data')

# %%

if __name__ == '__main__':
    
    freeze_support()       
        
    MLP = Static_MLP(dim_u=4,dim_out=1,dim_hidden=5,u_label=setpoints + ['T_wkz_0'],
                      y_label=['Durchmesser_innen'],name='MLP')
    
    data_norm = MLP.MinMaxScale(modelling_data)
    
    opt = ParamOptimizer(MLP,data_norm,data_norm,mode='static',
                        initializations=5,
                        res_path=target_h5.parents[0]/'models',
                        n_pool=5)
    
    results = opt.optimize()
    
    