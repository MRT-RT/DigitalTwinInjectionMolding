# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 12:10:56 2021

@author: alexa
"""

import h5py  
import os

import sys
sys.path.insert(0, "E:\GitHub\DigitalTwinInjectionMolding")
sys.path.insert(0, 'C:/Users/LocalAdmin/Documents/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, "/home/alexander/GitHub/DigitalTwinInjectionMolding/")

from DIM.miscellaneous.PreProcessing import hdf5_to_pd_dataframe_high_freq, add_csv_to_pd_dataframe

csv_filename = 'Versuchsplan_Stgrn.csv'

target_path = '/home/alexander/GitHub/DigitalTwinInjectionMolding/data/Stoergroessen/20220504/Versuchsplan/'

filenames = ['Versuchsplan_orig_Strgn.h5']


for filename in filenames:
    # Read h5 file
    file = h5py.File(filename,'r+')
    
    #convert and save as pd dataframe
    hdf5_to_pd_dataframe_high_freq(file,target_path)
    
cycle_files = os.listdir(target_path)


df_csv = pd.read_csv(csv_file_path,sep=';',index_col=0)

for cycle in cycle_files:
    
    df = pkl.load(open(target_path+cycle,'rb'))
    
    df = add_csv_to_pd_dataframe(df,df_csv)

# for i in range(1,251):
#     c = pkl.load(open('cycle'+str(i)+'.pkl','rb'))
#     diff = c.index.values[1::]-c.index.values[0:-1]
#     idx_del = np.where(diff<=0.001)
#     c.drop(index = c416.index[idx_del],inplace=True)
    # pkl.dump(c,open('cycle'+str(i)+'.pkl','wb'))
