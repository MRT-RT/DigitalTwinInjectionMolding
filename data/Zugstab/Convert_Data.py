# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 12:10:56 2021

@author: alexa
"""

import h5py  
import os
import pandas as pd
import pickle as pkl
import numpy as np

import sys
sys.path.insert(0, "E:\GitHub\DigitalTwinInjectionMolding")
sys.path.insert(0, 'C:/Users/LocalAdmin/Documents/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, "/home/alexander/GitHub/DigitalTwinInjectionMolding/")

from DIM.miscellaneous.PreProcessing import add_csv_to_pd_dataframe, hdf5_to_pd_dataframe

# pathe where to save data

target_path = '/home/alexander/GitHub/DigitalTwinInjectionMolding/data/Zugstab/data/'

''' Lege Versuchsplan im erforderlichen Format manuell an '''

col = ['Düsentemperatur', 'Werkzeugtemperatur',
       'Einspritzgeschwindigkeit', 'Umschaltpunkt', 'Nachdruckhöhe',
       'Nachdruckzeit', 'Staudruck', 'Kühlzeit', 'Gewicht', 'OK/N.i.O.',
       'Durchmesser_innen', 'Durchmesser_außen', 'Stegbreite_Gelenk',
       'Breite_Lasche', 'Rundheit_außen', 'Datum', 'Messzeit']


idx = pd.Series(data=range(1,327),name='Zyklusnummer')#list(range(1,327))

# Transcription from Versuchsplan_Zugstab.csv, 319 was added by AR because no
# Quality Data was collected for this part
idx_del = [36,43,44,45,54,57,60,82,83] + list(range(91,105)) + [111] + \
    list(range(145,151)) + [179] + list(range(192,201)) + \
    list(range(211,217)) +  list(range(227,236)) + \
    list(range(246,252)) +  list(range(262,268)) + \
    list(range(278,283)) + [319]   

# Initialize an empty pandas DataFrame
plan = pd.DataFrame(columns=col, index = idx)

# Load all data from csv
plan_csv = pd.read_csv('Versuchsplan_Zugstab.csv',delimiter=';')

quality_csv = pd.read_csv('Rauheit_Zugversuche.csv',delimiter=';',header=0,
                          skiprows=[1,2])
quality_csv =quality_csv.set_index('Nummer')
quality_csv.index.name = 'Zyklusnummer'

weight_csv = pd.read_csv('Gewicht_Zugstab.csv',delimiter=';',header=None)
weight_csv.index = idx
weight_csv.columns = ['Gewicht']

tickness_csv = pd.read_csv('Dicke_Zugstab.csv',delimiter=';',header=None)
tickness_csv.index = idx
tickness_csv.columns = ['Dicke']

# Transfer data from csv to DataFrame

lb = plan_csv['Start'].values
ub = plan_csv['Ende'].values

plan_data = plan_csv[['Düse', 'WKZ', 'Einspritzgeschw.', 'Umschaltpunkt']].values

for i in range(len(lb)):
    plan.loc[lb[i]:ub[i]+1, col[0:4] ] = plan_data[i,:]

plan['Gewicht'] =  weight_csv['Gewicht']
plan['Dicke'] =  tickness_csv['Dicke']

# Delete the parts for which no quality measurements were taken
plan = plan.drop(idx_del)

plan = pd.concat([plan,quality_csv],axis=1)

pkl.dump(plan,open(target_path+'Versuchsplan.pkl','wb'))

# Read process data from h5-file and save as pandas dataframes

filenames = ['Prozessdaten_Zugstab.h5']

for filename in filenames:

    # Read h5 file
    file = h5py.File(filename,'r+')
    
    #convert and save as pd dataframe
    hdf5_to_pd_dataframe(file,target_path)
       
for i in idx_del:
    os.remove(target_path+'cycle'+str(i)+'.pkl')
    
# add quality measurements to cycle data

for cycle in set(idx)-set(idx_del):
    
    cycle_path = target_path+'cycle'+str(cycle)+'.pkl'
    
    cycle_df = pkl.load(open(cycle_path,'rb'))
    
    cycle_num = cycle_df.loc[0]['cycle_num']
    
    for key in plan.keys():
        
        cycle_df[key]=np.nan
        cycle_df.loc[0][key] = plan.loc[cycle_num][key]

    # some measurements are constant trajectories    
    cycle_df['Werkzeugtemperatur'] = cycle_df.loc[0]['Werkzeugtemperatur']
    cycle_df['Düsentemperatur'] = cycle_df.loc[0]['Düsentemperatur']
    cycle_df['Einspritzgeschwindigkeit'] = cycle_df.loc[0]['Einspritzgeschwindigkeit']       

    cycle_df.rename(columns = {'Werkzeugtemperatur':'T_wkz_soll',
                         'Düsentemperatur':'T_nozz_soll',
                         'Einspritzgeschwindigkeit':'v_inj_soll'}, inplace = True)
    
    pkl.dump(cycle_df,open(cycle_path,'wb'))
    