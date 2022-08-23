# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 11:12:06 2022

@author: alexa
"""

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

from DIM.miscellaneous.PreProcessing import LoadFeatureData,MinMaxScale
from functions import estimate_polynomial
from DIM.models.model_structures import Static_MLP
from DIM.optim.param_optim import ModelTraining, static_mode

dim_hidden=10

data = pkl.load(open('data_doubleExp.pkl','rb'))
data_train = data['data_train']
data_test = data['data_test']


DoubleExpResults = pkl.load(open('Temp_Models.mdl','rb'))

setpoints = DoubleExpResults['setpoints']
Temp_Models = DoubleExpResults['Temp_Models']

# Model is supposed to map from setpoints to temperature models parameters
setpoints_lab = ['Düsentemperatur', 'Werkzeugtemperatur', 
                 'Einspritzgeschwindigkeit', 'Umschaltpunkt', 'Nachdruckhöhe',
                 'Nachdruckzeit', 'Staudruck','Kühlzeit']

param_lab =  list(Temp_Models[1].Parameters.keys())

# %% Arrange data accordingly
data_train = setpoints.copy()

data_train[param_lab] = None

for charge in data_train.index:
    params_charge =  Temp_Models[charge].Parameters
    
    # pd DataFrame takes no numpy arrays (iterables)
    params_charge = {key:[float(value)] for key,value in params_charge.items() }
    
    params_charge = pd.DataFrame.from_dict(params_charge)
    params_charge.index = [charge]
    params_charge.index.name= 'Charge'
    
    data_train.loc[charge,param_lab] = params_charge.loc[charge]

# %% Normalize data
data_train,scaler = MinMaxScale(data_train,setpoints_lab)

# %% Train model
setpoint_model = Static_MLP(dim_hidden=dim_hidden,dim_out=4,dim_u=8,
                            init_proc='xavier', name='setpt',
                            u_label=setpoints_lab,y_label=param_lab)

res = ModelTraining(model=setpoint_model,data_train=data_train,
                    data_val=data_train,initializations=10,
                    mode='static')

# Assign best parameters to model
setpoint_model.Parameters = \
    res.loc[pd.to_numeric(res['loss_val']).idxmin(),'params_val']

save = {'minmax_scaler':scaler,
        'Set_Model':setpoint_model,
        'param_est':res}

pkl.dump(save,open('setpoint_model_MLP_'+str(dim_hidden)+'.mdl','wb'))
# %% plot prediction

_,y_pred = static_mode(setpoint_model,data_train)


fig,ax = plt.subplots(1,1)
ax.plot(data_train[param_lab])
ax.plot(y_pred[param_lab])
ax.legend(param_lab)


