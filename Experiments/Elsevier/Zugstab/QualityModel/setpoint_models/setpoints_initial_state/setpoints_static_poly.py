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
sys.path.insert(0, 'E:/GitHub/DigitalTwinInjectionMolding/')

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from DIM.miscellaneous.PreProcessing import LoadSetpointData
from sklearn.preprocessing import PolynomialFeatures



charges = list(range(1,26))
split = 'all'

# targets = ['Durchmesser_innen']
# targets = ['E-Modul']
targets = ['Maximalspannung']

path = 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/data/Zugstab/data/normalized/'

data_train,data_val  = LoadSetpointData(path,charges,split)

# data = data_train.append(data_val)

inputs = ['Düsentemperatur', 'Werkzeugtemperatur', 'Einspritzgeschwindigkeit',
       'Umschaltpunkt']

# Try polynomial models up to order 10
for i in range(1,11):
    # Polynomial Model
    poly = PolynomialFeatures(i)
    X_poly_train = poly.fit_transform(data_train[inputs])
    X_poly_val = poly.transform(data_val[inputs])
    
    
    PolyModel = LinearRegression()
    PolyModel.fit(X_poly_train,data_train[targets])
    
    print(PolyModel.score(X_poly_val,data_val[targets]))
    


# Polynomial Model
# poly = PolynomialFeatures(4)
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


# # Linear Model Feaure Selection
# lin_reg = pd.DataFrame(columns=['BFR','feature_added'])

# for num_feat in range(1,9):
#     LinModel = LinearRegression()
#     if num_feat<8:
#         sfs = SequentialFeatureSelector(LinModel, n_features_to_select=num_feat,cv=50)
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

# # Polynomial Model feature selection
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