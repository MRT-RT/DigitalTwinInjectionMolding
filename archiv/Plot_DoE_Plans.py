#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 09:29:21 2022

@author: alexander
"""
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import sys
sys.path.insert(0, "E:\GitHub\DigitalTwinInjectionMolding")
sys.path.insert(0, 'C:/Users/LocalAdmin/Documents/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, "/home/alexander/GitHub/DigitalTwinInjectionMolding/")

from DIM.miscellaneous.PreProcessing import LoadFeatureData,LoadSetpointData
from DIM.models.model_structures import Static_MLP


path_sys = '/home/alexander/GitHub/DigitalTwinInjectionMolding'
charges = [list(range(1,26)),[1],[1],[1]]
split = 'all'


paths = ['/data/Stoergroessen/20220504/Versuchsplan/',
         '/data/Stoergroessen/20220504/Umschaltpkt_Stoerung/',
         '/data/Stoergroessen/20220505/T_wkz_Stoerung/',
         '/data/Stoergroessen/20220506/Rezyklat_Stoerung/']

groups = ['plan', 'Strg_V_um', 'Strg_T_wkz', 'Strg_Rezy']

data = []

for path,charge,group in zip(paths,charges,groups):
    data_1,data_2 = LoadFeatureData(path_sys+path,charge,split)
    data_conc = pd.concat([data_1,data_2])
    data_conc['Versuch']=group
    data.append(data_conc)
    
data = pd.concat(data)
data.index = range(len(data))    



# Plotte variierte Faktoren
factors = ['Düsentemperatur', 'Werkzeugtemperatur','Einspritzgeschwindigkeit',
           'Umschaltpunkt']

sns.pairplot(data[factors+['Versuch']],hue='Versuch')    

    
# Plotte resultierende Features
features  = ['T_wkz_0', 'T_wkz_max', 't_Twkz_max', 'T_wkz_int', 'p_wkz_max',
             'p_wkz_int', 'p_wkz_res', 't_pwkz_max', 'p_inj_int', 'p_inj_max']

sns.pairplot(data[(data['Versuch'] == 'plan') | (data['Versuch'] == 'Strg_V_um')   ][features+['Versuch']],hue='Versuch')    



# Plotte resultierendes Gewicht

sns.pairplot(data[['Gewicht']+['Versuch']],hue='Versuch')    
# LoadSetpointData(path,charges,split,y_label_q)

# # Normalize data
# data_train,minmax = MinMaxScale(data_train,u_label_q+y_label_q)
# data_val,_ = MinMaxScale(data_val,u_label_q+y_label_q,minmax)

# model_q = Static_MLP(dim_u=8, dim_out=1, dim_hidden=10,u_label=u_label_q,
#                     y_label=y_label_q,name='qual', init_proc='xavier')




path = '/home/alexander/GitHub/DigitalTwinInjectionMolding/data/Stoergroessen/20220504/'  # @home

# target_path = 'data/HighFrequencyMeasurements/'
# target_path = 'E:/GitHub/DigitalTwinInjectionMolding/data/HighFrequencyMeasurements/'
target_path = '/home/alexander/GitHub/DigitalTwinInjectionMolding/data/Stoergroessen/20220504/Versuchsplan/'