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
sys.path.insert(0, 'C:/Users/LocalAdmin/Documents/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, 'E:/GitHub/DigitalTwinInjectionMolding/')

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from DIM.miscellaneous.PreProcessing import LoadFeatureData
from sklearn.preprocessing import PolynomialFeatures



charges = list(range(1,26))
split = 'all'

# targets = ['Durchmesser_innen']
targets = ['Gewicht']


# path_sys = 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/'
# path_sys = '/home/alexander/GitHub/DigitalTwinInjectionMolding/' 
path_sys = 'E:/GitHub/DigitalTwinInjectionMolding/'

path = path_sys + '/data/Stoergroessen/20220504/Versuchsplan/normalized/'

data_train,data_val  = LoadFeatureData(path,charges,split)

# data = data_train.append(data_val)

# inputs = ['Düsentemperatur', 'Werkzeugtemperatur','Einspritzgeschwindigkeit',
#           'Umschaltpunkt','T_wkz_0','p_inj_0','x_0']

inputs = ['Düsentemperatur', 'Werkzeugtemperatur','Einspritzgeschwindigkeit',
          'Umschaltpunkt']

for i in range(1,11):
    # Polynomial Model
    poly = PolynomialFeatures(i)
    X_poly_train = poly.fit_transform(data_train[inputs])
    X_poly_val = poly.transform(data_val[inputs])
    
    
    PolyModel = LinearRegression()
    PolyModel.fit(X_poly_train,data_train[targets])
    
    print(PolyModel.score(X_poly_val,data_val[targets]))




# # Polynomial Model
# poly = PolynomialFeatures(2)
# X_poly_train = poly.fit_transform(data_train[inputs])
# X_poly_val = poly.transform(data_val[inputs])


# PolyModel = LinearRegression()
# PolyModel.fit(X_poly_train,data_train[targets])
# y=PolyModel.predict(X_poly_val)
# print(PolyModel.score(X_poly_val,data_val[targets]))

# data_val['y_est'] = y


# sns.stripplot(x=data_val.index,y=data_val[targets[0]],color='gray',size=8)
# sns.stripplot(x=data_val.index,y=data_val['y_est'],color='royalblue',size=8)
# # plt.xlim([1,30])

# plt.xticks([])
# plt.xlabel(None)
# plt.yticks([])
# plt.ylabel(None)


