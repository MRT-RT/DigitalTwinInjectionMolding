# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 13:48:30 2022

@author: alexa
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import pickle as pkl

import sys
from sklearn.model_selection import LeavePOut

path_dim = Path.cwd().parents[1]
sys.path.insert(0, path_dim.as_posix())

from DIM.miscellaneous.PreProcessing import LoadFeatureData
from functions import estimate_polynomial

# plt.close('all')

# path_data = path_dim / 'data/Versuchsplan/'

# Lade alle Daten
cross_val = False
normalize = True
degree = 3

data = pkl.load(open('data_minmax.pkl','rb'))

data_train = data['data_train']

inputs = ['T_wkz_0']
targets = ['Durchmesser_innen']

est_label = [label+'_est' for label in targets]
e_label = [label+'_error' for label in targets]

# %% Estimate coefficients of model

res_lpo = {}

lpo = LeavePOut(2)

for charge in set(data_train['Charge']):
    print(charge)
    res_lpo[charge] = {} 
    
    data_charge = data_train.loc[data_train['Charge']==charge]
    
    if normalize is True:
        idx_min = pd.to_numeric(data_charge['T_wkz_0']).idxmin()
        idx_min = pd.Series(idx_min).iloc[0]
        
        
        norm = data_charge.loc[idx_min]
        
        if type(norm) is pd.DataFrame:
            norm= norm.drop_duplicates().squeeze(axis=0)
            
        data_charge = data_charge - norm
        res_lpo[charge]['normalization'] = norm
        
    n_max = lpo.get_n_splits(data_charge)
    
    loss_min = np.inf
    count = 0
    
    if cross_val == True:
    
        for train_index, test_index in lpo.split(data_charge):
    
            res_charge = estimate_polynomial(degree,inputs,targets,
                                             data_charge.iloc[train_index],
                                             data_charge.iloc[test_index])
            
            count = count+1
            
            if res_charge['e_val']<loss_min:
                res_lpo[charge]['res']=res_charge
                loss_min = res_charge['e_val']
            # elif BFR_max == 0 and count==n_max:
            #     res_lpo[charge]['res']=res_charge
            
    else:
        
        res_charge = estimate_polynomial(degree,inputs,targets,
                                         data_charge,
                                         data_charge)
        
        res_lpo[charge]['res']=res_charge   


# %% Save Results in dataframe with cycles, columns, etc, ...

x_labels = ['a' + str(i) for i in  \
            range(0,res_lpo[1]['res']['model'].coef_.shape[1]) ]

data_train[x_labels]=np.nan

for charge in res_lpo.keys():
    data_train.loc[data_train['Charge']==charge,x_labels] = \
        res_lpo[charge]['res']['model'].coef_.flatten()

pkl.dump(data_train,open('Poly_d'+str(degree)+'_coef.pkl','wb'))
pkl.dump(res_lpo,open('Poly_d'+str(degree)+'_models.pkl','wb'))

# %% Plot prediction over cycles
             
fig,ax = plt.subplots(1,1)
plt.hist([float(res_lpo[key]['res']['BFR_val']) for key in res_lpo.keys()])


color_map = sns.color_palette()
fig,ax = plt.subplots(1,1)

fig.set_size_inches((15/2.54,4/2.54))

plt.tight_layout()

opts_true = {'linestyle':'None','marker':'d'}
opts_pred = {'linestyle':'None','marker':'o','color':color_map[5]}

for charge in set(data_train['Charge']):
        
    ax.plot(res_lpo[charge]['res']['predict_val'].index,
            res_lpo[charge]['res']['predict_val'][targets[0]],color='red',**opts_true)
    ax.plot(res_lpo[charge]['res']['predict_train'].index,
            res_lpo[charge]['res']['predict_train'][targets[0]],color='grey',**opts_true)
    
    ax.plot(res_lpo[charge]['res']['predict_val'].index,
            res_lpo[charge]['res']['predict_val'][est_label[0]],**opts_pred)
    ax.plot(res_lpo[charge]['res']['predict_train'].index,
            res_lpo[charge]['res']['predict_train'][est_label[0]],**opts_pred)
    

ax.set_xlim([0,100])#([1800,1900])
# ax.set_ylim([27.2,28])
ax.set_xlabel('$c$')
ax.set_ylabel('$D_{\mathrm{i}}$ in $\mathrm{mm}$')

# %% Plot coefficients

# Get coefficients

coefs = [res_lpo[charge]['res']['model'].coef_.flatten() for charge in res_lpo.keys()]
df = pd.DataFrame(data = coefs, columns =  x_labels)
# Define filter

filt = \
(\
# (data_train['Düsentemperatur'] == 0 ) &
(data_train['Werkzeugtemperatur'] == 0 ) & 
(data_train['Einspritzgeschwindigkeit'] == 0 ) &
(data_train['Umschaltpunkt'] == 0 ) &
(data_train['Nachdruckhöhe'] == 0 ) &
(data_train['Nachdruckzeit'] == 0 ) &
(data_train['Staudruck'] == 0 ) & 
(data_train['Kühlzeit'] == 0 )\
)

charge_idx = list(set(data_train.loc[filt]['Charge']))
x_label = set(data_train.loc[filt]['Düsentemperatur'])
# Plot coefficients for certain charges




# for charge in set(data_train['Charge']):
#     res_lpo[charge]=res_charge

fig,ax = plt.subplots(1,1)
# ax.plot(df,ax=ax,hue=df.index)




ax.plot(x_label,df.loc[charge_idx,x_labels[0]],marker='o')
ax.plot(x_label,df.loc[charge_idx,x_labels[1]],marker='o')
# ax.plot(idx,df.loc[idx,'T2'])



