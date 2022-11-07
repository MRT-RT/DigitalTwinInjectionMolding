# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 10:43:09 2022

@author: alexa
"""



from pathlib import Path
import sys
import h5py

path_dim = Path.cwd().parents[1]
sys.path.insert(0,path_dim.as_posix())


# import DigitalTwinFunctions as dtf
import time
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
import numpy as np
import seaborn as sns


from DIM.arburg470 import dt_functions as dtf

# %% Choose what to plot
q_label = 'Gewicht'

# %% Load new data and plot
source_h5 = Path('I:/Klute/DIM_Twin/DIM_20221104.h5')
target_h5 = Path('C:/Users/rehmer/Desktop/DIM/dm_Twkz.h5')

setpoints = ['v_inj_soll','V_um_soll','T_zyl5_soll','T_wkz_soll']   

dm = dtf.config_data_manager(source_h5,target_h5,setpoints)
# dm.get_cycle_data()

data_new = pd.read_hdf(dm.target_hdf5,'modelling_data')

fig,ax = plt.subplots(1,1)
sns.stripplot(data=data_new,x=data_new.index,y=q_label)


#%% Plot old data
# old_plan = pkl.load(open(path_dim/'data\Versuchsplan\Versuchsplan.pkl','rb'))
# fig,ax = plt.subplots(1,1)
# sns.stripplot(data=old_plan.loc[1:400],x=old_plan.loc[1:400].index,y=q_label)





