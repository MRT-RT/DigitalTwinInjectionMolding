#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 09:24:22 2022

@author: alexander
"""




import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

import sys
from sklearn.preprocessing import PolynomialFeatures


path_dim = Path.cwd().parents[1]
sys.path.insert(0, path_dim.as_posix())

from DIM.miscellaneous.PreProcessing import LoadFeatureData
from functions import estimate_polynomial

days = [list(range(1,652+1)), list(range(653,1302+1)), \
        list(range(1303,2052+1)),  list(range(2053,2782+1)) ]

path_data = path_dim / 'data/Versuchsplan/normalized/'

# Lade alle Daten

inputs = ['Düsentemperatur', 'Werkzeugtemperatur',
            'Einspritzgeschwindigkeit', 'Umschaltpunkt', 'Nachdruckhöhe',
            'Nachdruckzeit', 'Staudruck', 'Kühlzeit','T_wkz_0','p_inj_0',
            'x_0']
targets = ['Durchmesser_innen']

train_all,val_all  = LoadFeatureData(path_data.as_posix(),list(range(1,275)),
                                     'all', True)


# Prüfe ob alle Faktoren an jedem Tag mal variiert werden?

# Bilde Modell über alle Tage und für einzelne Tage
res_all = estimate_polynomial(4,inputs,targets,train_all,val_all)

# Vergleiche Residuen
fig1,ax1 = plt.subplots(2,2)

sns.scatterplot(data = res_all['predict_val'],x='y_true',y='y_est',ax=ax1[0,0])
sns.scatterplot(data = res_all['predict_val'],x='y_est',y='e',ax=ax1[0,1])
ax1[0,0].plot([27.25,27.9],[27.25,27.9])

# Klatsche Trainings- und Validierungsergebnisse zusammen und plotte nach Tagen

pred_all = pd.concat([res_all['predict_train'],res_all['predict_val']])

pred_all = res_all['predict_val']

pred_all.index.name = 'cycle'

for day in days:
    
    idx = [d for d in day if d in pred_all.index]
    
    # sns.scatterplot(data = pred_all.loc[idx],x='y_true',y='y_est',ax=ax1[1,0])
    # sns.scatterplot(data = pred_all.loc[idx],x='y_true',y='e',ax=ax1[1,1])

    sns.scatterplot(data = pred_all.loc[idx],x='cycle',y='y_est',ax=ax1[1,0])
    sns.scatterplot(data = pred_all.loc[idx],x='cycle',y='e',ax=ax1[1,1])
    
    print(pred_all.loc[idx].mean())


##############

data_all = pd.concat([train_all,val_all]).sort_index()

data_all['cycle']= data_all.index

charge1 = data_all.loc[11:20,['T_wkz_0','p_inj_0','x_0','Durchmesser_innen','cycle']]
charge6 = data_all.loc[51:60,['T_wkz_0','p_inj_0','x_0','Durchmesser_innen','cycle']]


fig2,ax2 = plt.subplots(2,1)


pd.plotting.parallel_coordinates(frame=charge6,
                                 class_column = 'cycle',
                                 colormap='Set1',
                                 ax=ax2[1])





clust_1 = train_all.loc[train_all['Durchmesser_innen']<27.4]
    
clust_2 = train_all.loc[(train_all['Durchmesser_innen']<27.55) &
                        (train_all['Durchmesser_innen']>27.44)]

