# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 15:16:22 2022

@author: LocalAdmin
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import numpy as np
sys.path.insert(0, "/home/alexander/GitHub/DigitalTwinInjectionMolding/")
sys.path.insert(0, 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, 'C:/Users/LocalAdmin/Documents/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, 'E:/GitHub/DigitalTwinInjectionMolding/')

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from DIM.miscellaneous.PreProcessing import LoadFeatureData
from DIM.optim.common import BestFitRate
from sklearn.preprocessing import PolynomialFeatures



charges = list(range(1,26))
split = 'all'
del_outl = True

targets = ['Gewicht']


# path_sys = 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/'
path_sys = 'C:/Users/LocalAdmin/Documents/GitHub/DigitalTwinInjectionMolding/'
# path_sys = '/home/alexander/GitHub/DigitalTwinInjectionMolding/' 
# path_sys = 'E:/GitHub/DigitalTwinInjectionMolding/'

path = path_sys + '/data/Stoergroessen/20220504/Versuchsplan/normalized/'

data_train,data_val  = LoadFeatureData(path,charges,split,del_outl)

# data = data_train.append(data_val)

inputs = ['Düsentemperatur', 'Werkzeugtemperatur', 'Einspritzgeschwindigkeit',
        'Umschaltpunkt']

# inputs = ['Düsentemperatur', 'Werkzeugtemperatur','Einspritzgeschwindigkeit',
#           'Umschaltpunkt','T_wkz_0','p_inj_0','x_0']

# Try polynomial models up to order 10
for i in [2]:#range(1,11):
    print('Polynomial order: ' + str(i))
    # Polynomial Model
    poly = PolynomialFeatures(i)
    X_poly_train = poly.fit_transform(data_train[inputs])
    X_poly_val = poly.transform(data_val[inputs])
    
    
    PolyModel = LinearRegression()
    PolyModel.fit(X_poly_train,data_train[targets])
    
    results_val = pd.DataFrame(data=[],columns=['y_true','y_est','e'],
                               index = data_val.index)
    
    results_val['y_true'] = data_val[targets]
    results_val['y_est'] = PolyModel.predict(X_poly_val)
    results_val['e'] = results_val['y_true']-results_val['y_est']
    
    print(PolyModel.score(X_poly_val,data_val[targets]))
    
    print('BFR: ' + str(BestFitRate(results_val['y_true'].values.reshape((-1,1)),
                results_val['y_est'].values.reshape((-1,1)))))
    
    e = abs(results_val['y_true']-results_val['y_est'])
    
    print('mean: '+ str(np.mean(e)))
    print('std: '+ str(np.std(e)))
    
    print(np.percentile(e,90))
    print(np.percentile(e,95))
    print(np.percentile(e,99))