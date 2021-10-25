# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 12:10:56 2021

@author: alexa
"""

import h5py  
import os

from DIM.miscellaneous.PreProcessing import hdf5_to_pd_dataframe, add_csv_to_pd_dataframe

csv_filename = 'Parameter_Qualitätsgrößen.csv'
        
# path = '/home/alexander/Downloads/Versuchsplan/' # @work
path = 'C:/Users/alexa/Downloads/Versuchsplan/'  # @home

target_path = 'data/Versuchsplan/'



filenames = ['Prozessgrößen_20211005.h5',
             'Prozessgrößen_20211006_1.h5',
             'Prozessgrößen_20211006_2.h5',
             'Prozessgrößen_20211007.h5',
             'Prozessgrößen_20211008.h5']


for filename in filenames:
    # Read h5 file
    file = h5py.File(path+filename,'r+')
    
    #convert and save as pd dataframe
    hdf5_to_pd_dataframe(file,target_path)
    
cycle_files = os.listdir(target_path)

for cycle in cycle_files:
    
    df = add_csv_to_pd_dataframe('data/Versuchsplan/cycle1.pkl',path+csv_filename)
    




