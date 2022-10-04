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

path_dim = Path.cwd()
sys.path.insert(0, path_dim.as_posix())

from DIM.miscellaneous.PreProcessing import PIM_Data


# source_hdf5 = 'C:\Users\alexa\Downloads\data\Prozessgrößen_20211005.h5'

# source_hdf5 = Path('C:/Users/klute/Documents/DIM/Versuchsplanung/Temperaturgang/OPC_UA_Arburg/Temperaturgangmessung-20221002.h5')


data_reader = pkl.load(open('Arburg_data_reader.pkl','rb'))


# Initialize plot
fig,ax = plt.subplots(2,2)
df_plot = pd.DataFrame(data=[],columns=['T_wkz_0','Durchmesser_innen','Gewicht'])
df_plot.index.rename = 'Zyklus'

while True:
        
    # Parse new data to target hdf5 if available
    data_reader.get_cycle_data()

    # Read data from target hdf5
    # df_overview = pd.read_hdf(data_reader.target_hdf5,key='overview')
    df_features = pd.read_hdf(data_reader.target_hdf5,key='features')   
    df_quality = pd.read_hdf(data_reader.target_hdf5,key='quality_meas')
    
    df_plot = pd.concat([df_features[['T_wkz_0']],
                         df_quality[['Durchmesser_innen','Gewicht']]],
                        axis=1)
    
        
    time.sleep(2)
    
    
# open taget file
# target_file = h5py.File(target_hdf5,'r')

# read from target file


# cycle_1756 = pd.read_hdf(target_hdf5,key='process_values/cycle_1756')
# overview = pd.read_hdf(target_hdf5,'overview')

