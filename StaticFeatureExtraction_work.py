# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 15:16:22 2022

@author: LocalAdmin
"""

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from DIM.miscellaneous.PreProcessing import StaticFeatureExtraction
from sklearn.preprocessing import PolynomialFeatures

charges = list(range(1,275))
# targets = ['Durchmesser_innen','Durchmesser_außen','Stegbreite_Gelenk','Gewicht',
#            'Stegbreite_Gelenk','Breite_Lasche']
targets = ['Durchmesser_innen']
# targets = ['Stegbreite_Gelenk']
# targets = ['Breite_Lasche']
# targets = ['Rundheit_außen']

data_train,data_val = StaticFeatureExtraction(charges,targets)

# data = data_train.append(data_val)

inputs = [col for col in data_train.columns if col not in targets]

# Normalize Data ?




# Linear Model
LinModel = LinearRegression()

# sfs = SequentialFeatureSelector(LinModel, n_features_to_select=11)
# sfs.fit(data[inputs], data[targets])
# inputs_sel = [inputs[i] for i in sfs.get_support(indices=True)]
# LinModel.fit(data[inputs_sel],data[targets])

LinModel.fit(data_train[inputs],data_train[targets])
print(LinModel.score(data_val[inputs],data_val[targets]))


# Polynomial Model
poly = PolynomialFeatures(2)
X_poly_train = poly.fit_transform(data_train[inputs])
X_poly_val = poly.fit_transform(data_val[inputs])


PolyModel = LinearRegression()
PolyModel.fit(X_poly_train,data_train[targets])

print(PolyModel.score(X_poly_val,data_val[targets]))


