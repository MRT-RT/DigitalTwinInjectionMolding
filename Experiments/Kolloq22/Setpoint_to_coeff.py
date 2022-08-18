# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 16:38:04 2022

@author: alexa
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import pickle as pkl

import sys
from sklearn.model_selection import KFold

path_dim = Path.cwd().parents[1]
sys.path.insert(0, path_dim.as_posix())

from DIM.miscellaneous.PreProcessing import LoadFeatureData,MinMaxScale
from functions import estimate_polynomial


# %% Estimate a model to predict coefficients

degree = 6

plt.close('all')

coef_data = pkl.load(open('Poly_d3_coef.pkl','rb'))
Temp_models = pkl.load(open('Poly_d3_models.pkl','rb'))

inputs = ['Düsentemperatur', 'Werkzeugtemperatur',
            'Einspritzgeschwindigkeit', 'Umschaltpunkt', 'Nachdruckhöhe',
            'Nachdruckzeit', 'Staudruck', 'Kühlzeit']

targets = ['a1','a2','a3']

coef_data = coef_data.groupby('Charge').first()

# coef_data,norm_coeff = MinMaxScale(coef_data,inputs+targets)

data_train = coef_data

res = estimate_polynomial(degree,inputs,targets,
                          data_train,
                          data_train)

print(res['BFR_train'])

# %% Evaluate whole model on test data

data = pkl.load(open('data_minmax.pkl','rb'))

data_train =  data['data_train']
data_test =  data['data_test']

data_all = pd.concat([data['data_train'], data['data_test']])

# Predict coeffcients first

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree)
X = poly.fit_transform(data_all[inputs])


coef_est = res['model'].predict(X)
coef_est = pd.DataFrame(data = coef_est, columns = ['a1','a2','a3'],
                        index = data_all.index)
# coef_est = MinMaxScale(coef_est,['a1','a2','a3'],
#                          minmax= norm_coeff,reverse = True)

data_all['a0'] = 0
data_all[['a1','a2','a3']] = coef_est[['a1','a2','a3']]

# Cheat: Rename Charge 144 to Charges 131 (they have the same setpoint)

data_all.loc[data_all['Charge']==144,'Charge'] = 131

# For each test datum, find appropriate model and predict Di

columns = ['T_wkz_0','Durchmesser_innen']

poly = PolynomialFeatures(3)


for cyc in data_all.index:
    
    io_data = data_all.loc[[cyc]]
    
    charge = io_data.loc[cyc,'Charge']
    model = Temp_models[charge]['res']['model']
    model.coef_ = io_data.loc[[cyc],['a0','a1','a2','a3']].values
    
    
    norm = Temp_models[charge]['normalization'][columns]

    io_data = io_data - norm
    
    # data_all.loc[cyc,'D_i_norm'] = io_data['Durchmesser_innen']
    
    X_test = poly.fit_transform(io_data['T_wkz_0'].values.reshape(-1,1))
    
    y_est = model.predict(X_test)
    
    y_est = y_est + norm['Durchmesser_innen']
    
    data_all.loc[cyc,'Durchmesser_innen_est'] = float(y_est)


color_map = sns.color_palette()
fig,ax = plt.subplots(1,1)

ax.plot(data_train.index, data_all.loc[data_train.index]['Durchmesser_innen'],
           color='grey',linestyle='None',marker='d')
ax.plot(data_test.index, data_all.loc[data_test.index]['Durchmesser_innen'],
           color='red',linestyle='None',marker='d')

ax.plot(data_all.index, data_all.loc[data_all.index]['Durchmesser_innen_est'],
           color=color_map[5],linestyle='None',marker='o')

ax.set_xlim([0,100])#([1800,1900])
ax.set_xlabel('$c$')
ax.set_ylabel('$D_{\mathrm{i}}$ in $\mathrm{mm}$')

