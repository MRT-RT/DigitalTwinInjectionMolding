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


# source_hdf5 = 'C:\Users\alexa\Downloads\data\Prozessgrößen_20211005.h5'

# source_hdf5 = Path('C:/Users/klute/Documents/DIM/Versuchsplanung/Temperaturgang/OPC_UA_Arburg/Temperaturgangmessung-20221002.h5')


data_reader = pkl.load(open('Arburg_data_reader.pkl','rb'))


# Initialize plot
plt.close('all')
fig,ax = plt.subplots(2,2)
df_plot = pd.DataFrame(data=[],columns=['T_wkz_0','Durchmesser_innen','Gewicht'])
df_plot.index.rename = 'Zyklus'

colormap = sns.color_palette(n_colors=10)

charges = np.vstack((0*np.ones((35,1),int),1*np.ones((31,1),int),2*np.ones((24,1),int),
                    3*np.ones((30,1),int),4*np.ones((30,1),int),5*np.ones((30,1),int),
                    6*np.ones((30,1),int),7*np.ones((30,1),int),8*np.ones((30,1),int),
                    9*np.ones((30,1),int)))

while True:
        
    # Parse new data to target hdf5 if available
    # try:
    data_reader.get_cycle_data()
    # except:
    time.sleep(5)
    # Read data from target hdf5
    # df_overview = pd.read_hdf(data_reader.target_hdf5,key='overview')
    df_features = pd.read_hdf(data_reader.target_hdf5,key='features')   
    df_quality = pd.read_hdf(data_reader.target_hdf5,key='quality_meas')
    
    df_features = df_features.sort_index()
    df_quality = df_quality.sort_index()
    
    df_plot = pd.concat([df_features[['T_wkz_0']],
                         df_quality[['Durchmesser_innen','Gewicht']]],
                        axis=1)
    

    
    df_plot['Charge'] = charges[0:len(df_plot)]
    
    df_plot = df_plot.sort_index()
    
    df_plot.groupby('Charge',as_index=False)
    
    df_plot['plot_index'] = None
    
    charges_list = list(set(df_plot['Charge']))
    
    for c in charges_list:
        
        idx_c = df_plot[df_plot['Charge']==c].index
        
        df_plot.loc[idx_c,'plot_index'] = np.array((range(0,len(idx_c))))
    
    
    
    sns.lineplot(data=df_plot,x = 'T_wkz_0', y = 'Durchmesser_innen', 
                 legend= False,
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
    
    
    
        
    time.sleep(1)
    plt.pause(20)
    
    
# open taget file
# target_file = h5py.File(target_hdf5,'r')

# read from target file


# cycle_1756 = pd.read_hdf(target_hdf5,key='process_values/cycle_1756')
# overview = pd.read_hdf(target_hdf5,'overview')

