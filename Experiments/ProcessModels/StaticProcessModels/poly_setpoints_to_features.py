# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 15:16:22 2022

@author: LocalAdmin
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
import sys
sys.path.insert(0, "/home/alexander/GitHub/DigitalTwinInjectionMolding/")
sys.path.insert(0, 'C:/Users/LocalAdmin/Documents/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, 'E:/GitHub/DigitalTwinInjectionMolding/')

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from DIM.miscellaneous.PreProcessing import LoadSetpointData,LoadFeatureData

from sklearn.preprocessing import PolynomialFeatures


''' Estimate a static polynomial model that maps process setpoints to resulting 
process measurement features '''

charges = list(range(1,275))
split = 'all'

inputs = ['Düsentemperatur',  'Werkzeugtemperatur',  'Einspritzgeschwindigkeit',
 'Umschaltpunkt','Nachdruckhöhe','Nachdruckzeit','Staudruck','Kühlzeit']

targets = ['T_wkz_0', 'T_wkz_max', 't_Twkz_max', 'T_wkz_int', 'p_wkz_max',
        'p_wkz_int', 'p_wkz_res', 't_pwkz_max']

# targets = ['T_wkz_max','p_wkz_max',
#         't_pwkz_max']

# targets = ['Gewicht']
# targets = ['Stegbreite_Gelenk']
# targets = ['Breite_Lasche']
# targets = ['Rundheit_außen']


# path = '/home/alexander/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
# path = 'C:/Users/LocalAdmin/Documents/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
path = 'E:/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'

data_train_set,data_val_set  = LoadSetpointData(path,charges,split,[])
data_train_feat,data_val_feat  = LoadFeatureData(path,charges,split,[])

data_train = pd.concat([data_train_set[inputs],data_train_feat[targets]],axis=1)
data_val = pd.concat([data_val_set[inputs],data_val_feat[targets]],axis=1)

inputs_save = []
R2_save = {}

for i in range(0,8):
   
   R2 = {}
    
   for input_try in inputs:
       # Polynomial Model
       inputs_sel = inputs_save + [input_try]
       
       R2_sel = {}
        
       for p in range(1,6):
           poly = PolynomialFeatures(p)
           X_poly_train = poly.fit_transform(data_train[inputs_sel])
           X_poly_val = poly.transform(data_val[inputs_sel])    
            
            
           PolyModel = LinearRegression()
           PolyModel.fit(X_poly_train,data_train[targets])
            
           R2_sel[p] = PolyModel.score(X_poly_val,data_val[targets])
            
            
       R2[input_try] = max(R2_sel.values())#max(R2, key=R2.get)
       
   inputs_save.append(max(R2, key=R2.get))
   
   R2_save[inputs_save[-1]] = max(R2.values())
   
   inputs.remove(inputs_save[-1])
   
   
print(R2_save)