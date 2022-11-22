# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 15:24:55 2022

@author: LocalAdmin
"""

# import multiprocessing
from multiprocessing import freeze_support
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
from DIM.models.models import Static_Multi_MLP
from DIM.optim.param_optim import ParamOptimizer

# Load DataManager specifically for this machine
dm = pkl.load(open('dm.pkl','rb'))

# source_h5 = Path('I:/Klute/DIM_Twin/DIM_20221101.h5')
# target_h5 = Path('C:/Users/rehmer/Desktop/DIM_Data/01_11_test.h5')

# source_h5 = Path('/home/alexander/Desktop/DIM/DIM_20221104.h5')
# target_h5 = Path('/home/alexander/Desktop/DIM/dm_Twkz.h5')

# setpoints = ['v_inj_soll','V_um_soll','T_wkz_soll']   

# Load DataManager specifically for this machine
# dm = dtf.config_data_manager(source_h5,target_h5,setpoints)

# Get data for model parameter estimation
# dm.get_cycle_data()
modelling_data = dm.get_modelling_data()
# target = ['Durchmesser_innen']

# %%



# if __name__ == '__main__':
    
#     freeze_support()       
    
#     inits = 20
    
#     for h in range(10,11):
#         for l in range(2,3):
    
#             name = 'Di_MLP_l'+str(l)+'_h'+str(h)
            
        
#             MLP = Static_Multi_MLP(dim_u=4,dim_out=1,dim_hidden=h,layers = l,
#                                    u_label=setpoints + ['T_wkz_0'],
#                                    y_label=target,name=name)
            
#             data_norm = MLP.MinMaxScale(modelling_data)
            
#             opt = ParamOptimizer(MLP,data_norm,data_norm,mode='static',
#                                 initializations=inits,
#                                 res_path=target_h5.parents[0]/name,
#                                 n_pool=20)
            
#             results = opt.optimize()
            
            # for i in range(inits):
            #     MLP = Static_MLP(dim_u=4,dim_out=1,dim_hidden=dim_hidden,u_label=setpoints + ['T_wkz_0'],
            #                   y_label=target,name='MLP')
                
            #     data_norm = MLP.MinMaxScale(modelling_data)
                
            #     MLP.Parameters = results.loc[i,'params_val']
            
            #     pkl.dump(MLP,open(model_path/('MLP'+str(i)+'.mod'),'wb'))
    
    