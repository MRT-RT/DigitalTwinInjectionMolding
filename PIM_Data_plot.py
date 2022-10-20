#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 10:02:48 2022

@author: alexander
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

path_dim = Path.cwd()
sys.path.insert(0, path_dim.as_posix())

from DIM.miscellaneous.PreProcessing import PIM_Data
import DigitalTwinFunctions as dtf

# source_hdf5 = 'C:\Users\alexa\Downloads\data\Prozessgrößen_20211005.h5'

# source_hdf5 = Path('C:/Users/klute/Documents/DIM/Versuchsplanung/Temperaturgang/OPC_UA_Arburg/Temperaturgangmessung-20221002.h5')


data_manager = dtf.config_data_manager()

# source_hdf5 = Path('C:/Users/LocalAdmin/Documents/DIM_Data/Messung 5.10/DIM_Temperaturgang_fixed.h5')
# source_hdf5 = Path('C:/Users/LocalAdmin/Documents/DIM_Data/Messung 6.10/Temperaturgangmessung-20221004.h5')
source_hdf5 = Path('C:/Users/LocalAdmin/Documents/DIM_Data/Messung 7.10/Temperaturgangmessung-20221005.h5')
# source_hdf5 = Path('C:/Users/LocalAdmin/Documents/DIM_Data/Messung 10.10/Temperaturgangmessung-20221010.h5')

# target_hdf5 = Path.cwd()/'TGang_051022.h5'
# target_hdf5 = Path.cwd()/'TGang_061022.h5'
target_hdf5 = Path.cwd()/'TGang_071022.h5'
# target_hdf5 = Path.cwd()/'TGang_101022.h5'


data_manager.source_hdf5 = source_hdf5
data_manager.target_hdf5 = target_hdf5


# Initialize plot
# plt.close('all')
fig,ax = plt.subplots(2,2)
df_plot = pd.DataFrame(data=[],columns=['T_wkz_0','Durchmesser_innen','Gewicht'])
df_plot.index.rename = 'Zyklus'

colormap = sns.color_palette(n_colors=10)



go = True

while go:
        
    # Parse new data to target hdf5 if available
    # try:
    data_manager.get_cycle_data()
    # df_ident = data_manager.get_ident_data()
    # except:
    # time.sleep(5)
    # Read data from target hdf5
    # df_overview = pd.read_hdf(data_reader.target_hdf5,key='overview')
    df_features = pd.read_hdf(data_manager.target_hdf5,key='features')   
    df_quality = pd.read_hdf(data_manager.target_hdf5,key='quality_meas')
    
    df_features = df_features.sort_index()
    df_quality = df_quality.sort_index()
    
    df_plot = pd.concat([df_features[['T_wkz_0']],
                          df_quality[['Durchmesser_innen','Gewicht']]],
                        axis=1)
    
    df_plot = df_plot.sort_index()
    
    df_plot['timedelta'] = df_quality['Messzeit'].diff()
    df_plot['Charge'] = 0
    new_charge_idx = df_plot.index[df_plot['timedelta']>pd.Timedelta(4,'m')]
    
    for idx in new_charge_idx:
        df_plot.loc[6:idx-1,'Charge'] = df_plot.loc[6:idx-1,'Charge']+1
        

    df_plot.groupby('Charge',as_index=False)
    
    df_plot['plot_index'] = None
    
    charges_list = list(set(df_plot['Charge']))
    
    for c in charges_list:
        
        idx_c = df_plot[df_plot['Charge']==c].index
        
        df_plot.loc[idx_c,'plot_index'] = np.array((range(0,len(idx_c))))
    
    
    
    sns.lineplot(data=df_plot,x = 'T_wkz_0', y = 'Durchmesser_innen', 
                  legend= True,
                  hue = 'Charge',marker='o', markersize= 15,ax = ax[0,0],
                  palette = colormap[0:len(charges_list)])
    
    
    sns.lineplot(data=df_plot,x = 'T_wkz_0', y = 'Gewicht', hue = 'Charge',
                  marker='o', markersize= 15, legend=False,ax = ax[0,1],
                  palette = colormap[0:len(charges_list)])
    
    sns.lineplot(data=df_plot,x = 'plot_index', y = 'Durchmesser_innen',
                  hue = 'Charge',
                  marker='o', markersize= 15,legend=False,ax = ax[1,0],
                  palette = colormap[0:len(charges_list)])
     
    sns.lineplot(data=df_plot,x = 'plot_index', y = 'Gewicht', 
                  hue = 'Charge',
                  marker='o', markersize= 15,legend=False,ax = ax[1,1],
                  palette = colormap[0:len(charges_list)])
    

    
    go = False    
    # time.sleep(1)
    # plt.pause(20)
    
    
    

ax[0,0].set_ylim([27.8,28])
ax[0,1].set_ylim([8.1,8.24])

ax[1,0].set_ylim([27.8,28])
ax[1,1].set_ylim([8.1,8.24])
