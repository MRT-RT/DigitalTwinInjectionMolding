# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 09:08:22 2022

@author: LocalAdmin
"""

from pathlib import Path
import sys
import h5py
import pandas as pd
import pickle as pkl
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

path_dim = Path.cwd().parents[1]
sys.path.insert(0, path_dim.as_posix())

from DIM.miscellaneous.PreProcessing import LoadFeatureData

path_data = path_dim /'data/Stoergroessen/20220504/Versuchsplan'


data_train,data_val = LoadFeatureData(path_data.as_posix(),range(1,26),'all',True)

data = pd.concat([data_train,data_val],axis=0)

plt.close('all')

fig,ax = plt.subplots(5,5)
ax = ax.flatten()

for charge in range(1,26):
    data_charge = data.loc[data['Charge']==charge]
    ax[charge-1].plot(data_charge['T_wkz_0'],data_charge['Durchmesser_innen'],
                      linestyle='none',marker='o')
    ax[charge-1].set_xlim([34,60])
    ax[charge-1].set_ylim([26.3,28.5])
    
    ax[charge-1].set_title('Charge'+str(charge))
    