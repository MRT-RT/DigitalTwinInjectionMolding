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

from DIM.miscellaneous.PreProcessing import add_csv_to_pd_dataframe, hdf5_to_pd_dataframe

csv_filename = 'Strgrsn_T_wkz.csv'
filenames = ['Prozessgrößen_Strgn_WKZ_Temp.h5']


# target_path = '/home/alexander/GitHub/DigitalTwinInjectionMolding/data/Stoergroessen/20220505/T_wkz_Stoerung/'
target_path ='E:/GitHub/DigitalTwinInjectionMolding/data/Stoergroessen/20220505/T_wkz_Stoerung/'

for filename in filenames:

    # Read h5 file
    file = h5py.File(filename,'r+')
    
    #convert and save as pd dataframe
    hdf5_to_pd_dataframe(file,target_path)
    
    
    
os.remove(target_path+'cycle40.pkl')                                           # Teil verloren
os.remove(target_path+'cycle105.pkl')                                          # Teil verloren




cycle_files = os.listdir(target_path)

csv = pd.read_csv(csv_filename,delimiter=';')  

for cycle in cycle_files:

    c = pkl.load(open(cycle,'rb'))    

    
    df = add_csv_to_pd_dataframe(c,csv_filename)

# for i in range(1,251):
#     c = pkl.load(open('cycle'+str(i)+'.pkl','rb'))
#     diff = c.index.values[1::]-c.index.values[0:-1]
#     idx_del = np.where(diff<=0.001)
#     c.drop(index = c416.index[idx_del],inplace=True)
    # pkl.dump(c,open('cycle'+str(i)+'.pkl','wb'))
