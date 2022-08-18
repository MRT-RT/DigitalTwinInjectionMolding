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

# plt.close('all')

days = [list(range(1,652+1)), list(range(653,1302+1)), \
        list(range(1303,2052+1)),  list(range(2053,2782+1)) ]

path_data = path_dim / 'data/Versuchsplan/normalized/'

# Lade alle Daten

inputs = ['Düsentemperatur', 'Werkzeugtemperatur',
            'Einspritzgeschwindigkeit', 'Umschaltpunkt', 'Nachdruckhöhe',
            'Nachdruckzeit', 'Staudruck', 'Kühlzeit','T_wkz_0','p_inj_0',
            'x_0']
targets = ['Durchmesser_innen']

est_label = [label+'_est' for label in targets]
e_label = [label+'_error' for label in targets]

train_all,val_all  = LoadFeatureData(path_data.as_posix(),list(range(1,275)),
                                     'all', True)


# Prüfe ob alle Faktoren an jedem Tag mal variiert werden?

# Bilde Modell über alle Tage und für einzelne Tage
res_all = estimate_polynomial(4,inputs,targets,train_all,val_all)

# Vergleiche Residuen
fig1,ax1 = plt.subplots(2,2)

sns.scatterplot(data = res_all['predict_val'],x=targets[0],y=est_label[0],ax=ax1[0,0])
sns.scatterplot(data = res_all['predict_val'],x=targets[0],y=e_label[0],ax=ax1[0,1])
ax1[0,0].plot([27.25,27.9],[27.25,27.9])

# Klatsche Trainings- und Validierungsergebnisse zusammen und plotte nach Tagen

pred_all = pd.concat([res_all['predict_train'],res_all['predict_val']])

pred_all = res_all['predict_val']

pred_all.index.name = 'cycle'

for day in days:
    
    idx = [d for d in day if d in pred_all.index]
    
    # sns.scatterplot(data = pred_all.loc[idx],x='y_true',y='y_est',ax=ax1[1,0])
    # sns.scatterplot(data = pred_all.loc[idx],x='y_true',y='e',ax=ax1[1,1])

    sns.scatterplot(data = pred_all.loc[idx],x='cycle',y=est_label[0],ax=ax1[1,0])
    sns.scatterplot(data = pred_all.loc[idx],x='cycle',y=e_label[0],ax=ax1[1,1])
    
    print(pred_all.loc[idx].mean())


##############

data_all = pd.concat([train_all,val_all]).sort_index()

data_all['cycle']= data_all.index

charge1 = data_all.loc[11:20,['T_wkz_0','p_inj_0','x_0','Durchmesser_innen','cycle']]
charge6 = data_all.loc[51:60,['T_wkz_0','p_inj_0','x_0','Durchmesser_innen','cycle']]


fig2,ax2 = plt.subplots(2,1)


pd.plotting.parallel_coordinates(frame=charge1,
                                 class_column = 'cycle',
                                 colormap='Set1',
                                 ax=ax2[0])

pd.plotting.parallel_coordinates(frame=charge6,
                                 class_column = 'cycle',
                                 colormap='Set1',
                                 ax=ax2[1])

################

# Di for each charge over cavity temperature

fig3,ax3 = plt.subplots(5,5)

counter = 0
for charge in set(data_all['Charge']): # [131,144]:#
    
    data_plot = data_all.loc[data_all['Charge']==charge]
    
    ax3.flat[counter].plot(data_plot['T_wkz_0'],
                           data_plot['Durchmesser_innen'],
                           linestyle='None', marker='o')
    
    counter = counter + 1
    



# estimate polynomial for every single setpoint and take a look at coefficients




fig3,ax3 = plt.subplots(1,1)

plt.hist([float(result['BFR_val']) for result in res])
plt.hist([float(result['BFR_train']) for result in res])



# %%
# plt.close('all')
# fig4,ax4 = plt.subplots(1,1)
x = np.arange(0,0.2,1/1000)

T1 = 12.82861547485901
T2 = 2*T1
a = 0.08236384898638958
b = 0.19759018643495319

D = a*np.exp(-T1*(x)) - b*np.exp(-T2*(x))
ax4.plot(x,D)

# ax4.plot(x,np.exp(-0.1*(x)))
# ax4.plot(x,-np.exp(-1.0*(x)))


