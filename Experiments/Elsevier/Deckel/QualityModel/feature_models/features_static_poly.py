# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 15:16:22 2022

@author: LocalAdmin
"""
import sys
sys.path.insert(0, "/home/alexander/GitHub/DigitalTwinInjectionMolding/")
sys.path.insert(0, 'C:/Users/LocalAdmin/Documents/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, 'E:/GitHub/DigitalTwinInjectionMolding/')

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from DIM.miscellaneous.PreProcessing import LoadFeatureData
from sklearn.preprocessing import PolynomialFeatures



charges = list(range(1,26))
split = 'all'


# targets = ['Durchmesser_innen']
targets = ['Gewicht']

path = '/home/alexander/GitHub/DigitalTwinInjectionMolding/data/Stoergroessen/20220504/Versuchsplan/normalized/'



data_train,data_val  = LoadFeatureData(path,charges,split,targets)

# data = data_train.append(data_val)

inputs = [col for col in data_train.columns if col not in targets]

inputs = inputs[0:8]
# Normalize Data ?

for i in range(1,11):
    # Polynomial Model
    poly = PolynomialFeatures(i)
    X_poly_train = poly.fit_transform(data_train[inputs])
    X_poly_val = poly.fit_transform(data_val[inputs])
    
    
    PolyModel = LinearRegression()
    PolyModel.fit(X_poly_train,data_train[targets])
    
    print(PolyModel.score(X_poly_val,data_val[targets]))


# Linear Model
# LinModel = LinearRegression()

# # sfs = SequentialFeatureSelector(LinModel, n_features_to_select=10)
# # sfs.fit(data_train[inputs], data_train[targets])
# # inputs = [inputs[i] for i in sfs.get_support(indices=True)]

# # LinModel.fit(data[inputs],data[targets])
# LinModel.fit(data_train[inputs],data_train[targets])
# print(LinModel.score(data_val[inputs],data_val[targets]))


# # Polynomial Model
# poly = PolynomialFeatures(8)
# X_poly_train = poly.fit_transform(data_train[inputs])
# X_poly_val = poly.fit_transform(data_val[inputs])


# PolyModel = LinearRegression()
# PolyModel.fit(X_poly_train,data_train[targets])

# print(PolyModel.score(X_poly_val,data_val[targets]))


