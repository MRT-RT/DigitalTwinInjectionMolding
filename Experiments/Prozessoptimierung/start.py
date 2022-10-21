# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 15:16:22 2022

@author: LocalAdmin
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

path_dim = Path.cwd().parents[1]
sys.path.insert(0, path_dim.as_posix())

from DIM.miscellaneous.PreProcessing import LoadFeatureData, MinMaxScale
from DIM.models.model_structures import Static_Multi_MLP
from DIM.optim.param_optim import ParamOptimizer
from DIM.optim.control_optim import StaticProcessOptimizer

import multiprocessing

import pickle as pkl

# %% Load Data and identified model

res = pkl.load(open('MLP_2layers_T0/QM_MLP_Di_h10.pkl','rb'))
model = res['model']


# look for best parameters
idx = res['est_params']['loss_val'].idxmin()
model.Parameters = res['est_params'].loc[idx,'params_val']

data = pkl.load(open('data.pkl','rb'))

data_train = data['data_train']
data_test = data['data_test']

# %%
plt.close('all')
fig1,ax1 = plt.subplots(1,1)
opts = {'marker':'o','linestyle':'None'}
ax1.plot(data_train.index,data_train['Durchmesser_innen'],
             color='grey',**opts)
ax1.plot(data_test.index,data_test['Durchmesser_innen'],
             color='red',**opts)
ax1.set_xlim([0,100])
ax1.set_xlabel('Zyklus')
ax1.set_ylabel('Durchmesser_innen in mm')

# %%

data_train_norm = model.MinMaxScale(data_train)
data_test_norm = model.MinMaxScale(data_test)

# ParamOptimizer = ParamOptimizer(model,data_train_norm,data_test_norm,5,
#                                 mode='static',n_pool=5)

if __name__ == '__main__':
    
    multiprocessing.freeze_support()
    
    # ResOptim = ParamOptimizer.optimize()
    
    # idx = ResOptim['loss_val'].idxmin()
    # model.Parameters = ResOptim.loc[idx,'params_val']
    
    loss, pred = model.static_mode(data_test_norm)
    pred_un = model.MinMaxScale(pred,reverse=True)
    
    # %% Plot prediction
    ax1.plot(pred_un.index,pred_un['Durchmesser_innen'],
                 color='blue',**opts)
    
    
    # %% Find setpoint for 27.4 mm D_i
    fix_inputs =  pd.DataFrame.from_dict({'T_wkz_0': [57.5]})
    Q_target =  pd.DataFrame.from_dict({'Durchmesser_innen': [27.4]})
    
    input_init = pd.DataFrame.from_dict({'Düsentemperatur': [250.0],
                                         'Werkzeugtemperatur': [40.0],
                                         'Einspritzgeschwindigkeit':[40.0], 
                                         'Umschaltpunkt':[13.5], 
                                         'Nachdruckhöhe':[550],
                                         'Nachdruckzeit':[5], 
                                         'Staudruck':[50.0], 
                                         'Kühlzeit':[20.0]})
    
    # get constraints from data
    constraints = []
    
    for u in set(model.u_label)-set(['T_wkz_0']):
        constraints.append((u,'>'+str(data_train_norm[u].min())))
        constraints.append((u,'<'+str(data_train_norm[u].max())))   
    
    
    fix_inputs_norm  = model.MinMaxScale(fix_inputs)
    Q_target_norm = model.MinMaxScale(Q_target)
    input_init_norm = model.MinMaxScale(input_init)
    
    Optimizer = StaticProcessOptimizer(model=model)
    U_sol = Optimizer.optimize(Q_target_norm,fix_inputs_norm,
                                input_init=input_init_norm,
                                constraints=constraints)
    
    U_sol_norm = model.MinMaxScale(U_sol,reverse=True)

    
    # %% Check if at least model predicts desired quality at this setpoint
    loss,test = model.static_mode(U_sol)
   
    test_un = model.MinMaxScale(test,reverse=True)









