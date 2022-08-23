# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 17:33:50 2022

@author: LocalAdmin
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import pickle as pkl

import sys


path_dim = Path.cwd().parents[1]
sys.path.insert(0, path_dim.as_posix())

from DIM.miscellaneous.PreProcessing import LoadFeatureData, MinMaxScale
from functions import estimate_polynomial
from DIM.models.model_structures import Static_MLP
from DIM.models.injection_molding import staticQualityModel
from DIM.optim.param_optim import ModelTraining, static_mode

data = pkl.load(open('data_doubleExp.pkl','rb'))
data_train = data['data_train']
data_test = data['data_test']


# Load Temperature model
DoubleExpResults = pkl.load(open('Temp_Models.mdl','rb'))

setpoints = DoubleExpResults['setpoints']
Temp_Models = DoubleExpResults['Temp_Models']


# Load Setpoint Model
SetpointResults = pkl.load(open('setpoint_model_MLP_10.mdl','rb'))

minmax_scaler = SetpointResults['minmax_scaler']
Set_Model = SetpointResults['Set_Model']
param_est = SetpointResults['param_est']

setpoints_lab = ['Düsentemperatur', 'Werkzeugtemperatur', 
                 'Einspritzgeschwindigkeit', 'Umschaltpunkt', 'Nachdruckhöhe',
                 'Nachdruckzeit', 'Staudruck','Kühlzeit']
T = ['T_wkz_0']
Di = ['Durchmesser_innen']

# Scale setpoint according to when MLP was trained
setpoints,_ = MinMaxScale(setpoints,setpoints_lab,minmax=minmax_scaler)
data_train,_ = MinMaxScale(data_train,setpoints_lab,minmax=minmax_scaler)
data_test,_ = MinMaxScale(data_test,setpoints_lab,minmax=minmax_scaler)

QM = staticQualityModel(Set_Model,Temp_Models,setpoints,'QM')

QM_opt_params = []

for init in param_est.index:

    QM.setpoint_model.Parameters = param_est.loc[init,'params_val']
    
    # Optimize parameters
    res = ModelTraining(QM,data_train,data_test, initializations=1,mode='static')
    res.index = [init]
    
    QM_opt_params.append(res)
    
QM_opt_params = pd.concat(QM_opt_params)

# Save results
save = {'param_est':QM_opt_params}
pkl.dump(save,open('MLP_DoubleExp_QModel','wb'))

# # Assign optimal parameters
QM.setpoint_model.Parameters = \
QM_opt_params.loc[QM_opt_params['loss_val'].idxmin(),'params_val']


# %% Plot predictions

# QM.setpoint_model.Parameters = results.loc[0,'params_val']
_,pred_train = static_mode(QM,data_train)


fig,ax = plt.subplots(1,1)
ax.plot(data_train.index.values,data_train[Di].values,
                            linestyle='None', marker='o')  
ax.plot(pred_train.index.values,pred_train[Di].values,
                            linestyle='None', marker='x')  








