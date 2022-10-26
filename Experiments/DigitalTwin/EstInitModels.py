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
source_hdf = Path('C:/Users/LocalAdmin/Documents/DIM_Data/Messung 5.10/hist_data.h5')
target_hdf = Path.cwd()/'all_data_05_10_22.h5'
# target_hdf = Path.cwd()/'DOE_2_dm.h5'


# Load DataManager specifically for this machine
dm = dtf.config_data_manager(source_hdf,target_hdf,['v_inj_soll','V_um_soll'])

# Get data for model parameter estimation
modelling_data = pd.read_hdf(dm.target_hdf5, 'modelling_data')


if __name__ == '__main__':
    
    freeze_support()       
        
    MLP = Static_MLP(dim_u=2,dim_out=1,dim_hidden=5,u_label=['T_wkz_0','V_um_soll'],
                      y_label=['Durchmesser_innen'],name='MLP')
    
    data_norm = MLP.MinMaxScale(modelling_data)
    
    opt = ParamOptimizer(MLP,data_norm,data_norm,mode='static',
                        initializations=5,res_path=Path.cwd()/'results',
                        n_pool=5)
    
    results = opt.optimize()
            
        # idx = res['loss_val'].idxmin()
        
        # MLP.Parameters = res.loc[idx,'params_val']
        
        # pkl.dump(MLP,open('Models/'+MLP.name+'.mod','wb'))
    
    