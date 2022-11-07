# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 09:08:48 2022

@author: rehmer
"""

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

# %%
res = pkl.load(open('C:/Users/rehmer/Desktop/DIM_Data/Di_MLP_l2_h10/overview.pkl','rb'))
models = pkl.load(open('C:/Users/rehmer/Desktop/DIM_Data/Da_MLP_l2_h10/models.pkl','rb'))
model = models[6]['val']  

target = 'Durchmesser_außen'       

# %%

source_h5 = Path('I:/Klute/DIM_Twin/DIM_20221101.h5')
target_h5 = Path('C:/Users/rehmer/Desktop/DIM_Data/01_11_test.h5')

setpoints = ['v_inj_soll','V_um_soll','T_zyl5_soll']                           # T_wkz_soll fehlt

# Load DataManager specifically for this machine
dm = dtf.config_data_manager(source_h5,target_h5,setpoints)

# %%

data = pd.read_hdf(dm.target_hdf5,'modelling_data')
data_norm = model.MinMaxScale(data)

# %%
pred_norm = model.static_mode(data_norm)

pred = model.MinMaxScale(pred_norm[1],reverse=True)

# %%
plt.figure()
sns.stripplot(data=data,y=target,x=data.index)
sns.stripplot(data=pred,y=target,x=pred.index,color='grey')
