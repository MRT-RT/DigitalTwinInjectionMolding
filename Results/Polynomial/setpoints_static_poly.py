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
from DIM.miscellaneous.PreProcessing import LoadSetpointData
from sklearn.preprocessing import PolynomialFeatures



charges = list(range(1,275))
# targets = ['Durchmesser_innen','Durchmesser_außen','Stegbreite_Gelenk','Gewicht',
#            'Stegbreite_Gelenk','Breite_Lasche']
targets = ['Durchmesser_innen']
# targets = ['Stegbreite_Gelenk']
# targets = ['Breite_Lasche']
# targets = ['Rundheit_außen']


# path = '/home/alexander/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
# path = 'C:/Users/LocalAdmin/Documents/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
path = 'E:/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'

data_train,data_val,_,_,_,_  = LoadSetpointData(path,charges,targets)

# data = data_train.append(data_val)

inputs = [col for col in data_train.columns if col not in targets]




# LinModel.fit(data_train[inputs],data_train[targets])
# print(LinModel.score(data_val[inputs],data_val[targets]))


for i in range(1,3):
    # Polynomial Model
    poly = PolynomialFeatures(i)
    X_poly_train = poly.fit_transform(data_train[inputs])
    X_poly_val = poly.fit_transform(data_val[inputs])
    
    
    PolyModel = LinearRegression()
    PolyModel.fit(X_poly_train,data_train[targets])
    
    print(PolyModel.score(X_poly_val,data_val[targets]))

# Linear Model Feaure Selection
# lin_reg = pd.DataFrame(columns=['BFR','feature_added'])

# for num_feat in range(1,9):
#     LinModel = LinearRegression()
#     if num_feat<8:
#         sfs = SequentialFeatureSelector(LinModel, n_features_to_select=num_feat,cv=100)
#         sfs.fit(data_train[inputs], data_train[targets])
#         inputs_step = [inputs[i] for i in sfs.get_support(indices=True)]
#     else:
#         inputs_step = inputs
        
#     LinModel.fit(data_train[inputs_step],data_train[targets])
    
#     R2 = LinModel.score(data_val[inputs_step],data_val[targets])

#     if num_feat>1:
#         # print(list(lin_reg['feature_added']))
#         new_feat = [feat for feat in inputs_step if feat not in list(lin_reg['feature_added'])][0]
#         # print(new_feat)
#     else:
#         new_feat = inputs_step[0]
        
        
#     lin_reg=lin_reg.append({'BFR':R2,'feature_added':new_feat},ignore_index=True)


# fig, ax = plt.subplots()
# sns.barplot(x="feature_added", y="BFR", data=lin_reg,ax=ax)
# ax.set_title(targets)

# Polynomial Model feature selection
# pol_reg = pd.DataFrame(columns=['BFR','feature_added'])

# poly = PolynomialFeatures(2)
# X_poly_train = poly.fit_transform(data_train[inputs])
# X_poly_val = poly.transform(data_val[inputs])

# inputs_poly = poly.get_feature_names(inputs)

# data_train_poly = pd.DataFrame(data=X_poly_train,columns=[inputs_poly],index=data_train.index)
# data_train_poly[targets] = data_train[targets]

# data_val_poly = pd.DataFrame(data=X_poly_val,columns=[inputs_poly],index=data_val.index)
# data_val_poly[targets] = data_val[targets]

# for num_feat in range(1,9):

    
#     PolyModel = LinearRegression()
#     if num_feat<8:
#         sfs = SequentialFeatureSelector(PolyModel, n_features_to_select=num_feat,cv=100)
#         sfs.fit(data_train_poly[inputs_poly], data_train_poly[targets])
#         inputs_step = [inputs_poly[i] for i in sfs.get_support(indices=True)]
#     else:
#         inputs_step = inputs_poly
        
#     PolyModel.fit(data_train_poly[inputs_step],data_train_poly[targets])
    
#     R2 = PolyModel.score(data_val_poly[inputs_step],data_val_poly[targets])

#     if num_feat>1:
#         # print(list(lin_reg['feature_added']))
#         new_feat = [feat for feat in inputs_step if feat not in list(lin_reg['feature_added'])][0]
#         # print(new_feat)
#     else:
#         new_feat = inputs_step[0]
        
        
#     pol_reg=lin_reg.append({'BFR':R2,'feature_added':new_feat},ignore_index=True)


# fig, ax = plt.subplots()
# sns.barplot(x="feature_added", y="BFR", data=pol_reg,ax=ax)
# ax.set_title(targets)