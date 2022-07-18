# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 15:16:22 2022

@author: LocalAdmin
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.insert(0, "/home/alexander/GitHub/DigitalTwinInjectionMolding/")
sys.path.insert(0, 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, 'C:/Users/LocalAdmin/Documents/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, 'E:/GitHub/DigitalTwinInjectionMolding/')

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from DIM.miscellaneous.PreProcessing import LoadFeatureData
from sklearn.preprocessing import PolynomialFeatures



charges = list(range(1,26))
split = 'all'
del_outl = True

targets = ['Gewicht']


# path_sys = 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/'
path_sys = 'C:/Users/LocalAdmin/Documents/GitHub/DigitalTwinInjectionMolding/'
# path_sys = '/home/alexander/GitHub/DigitalTwinInjectionMolding/' 
# path_sys = 'E:/GitHub/DigitalTwinInjectionMolding/'

path_train = path_sys + '/data/Stoergroessen/20220504/Versuchsplan/normalized/'
path_dist = path_sys + '/data/Stoergroessen/20220504/Umschaltpkt_Stoerung/normalized/'


data_train,data_val  = LoadFeatureData(path_train,charges,split,True)
data_st1,data_st2  = LoadFeatureData(path_dist,[1],split,False)

data_st = pd.concat([data_st1,data_st2])

# data = data_train.append(data_val)

inputs = ['Düsentemperatur', 'Werkzeugtemperatur', 'Einspritzgeschwindigkeit',
        'Umschaltpunkt']

# inputs = ['Düsentemperatur', 'Werkzeugtemperatur','Einspritzgeschwindigkeit',
#           'Umschaltpunkt','T_wkz_0','p_inj_0','x_0']

# Try polynomial models up to order 10
for i in range(1,11):
    # Polynomial Model
    poly = PolynomialFeatures(i)
    X_poly_train = poly.fit_transform(data_train[inputs])
    X_poly_val = poly.transform(data_st[inputs])
    
    
    PolyModel = LinearRegression()
    PolyModel.fit(X_poly_train,data_train[targets])
    
    print(PolyModel.score(X_poly_val,data_st[targets]))
    
