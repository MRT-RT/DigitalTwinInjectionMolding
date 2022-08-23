# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 10:38:18 2022

@author: alexa
"""

# import h5py
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
from DIM.models.model_structures import DoubleExponential
from DIM.optim.param_optim import ModelTraining, static_mode

# plt.close('all')

# path_data = path_dim / 'data/Versuchsplan/'

# Lade alle Daten
cross_val = False
normalize = True

# charge_ = 25

data = pkl.load(open('data_doubleExp.pkl','rb'))

data_train = data['data_train']
data_test = data['data_test']

# data_train=data_train.loc[data_train['Charge']==charge_]
# data_test=data_test.loc[data_test['Charge']==charge_]

# data_all = pd.concat([data_train,data_test])

T = ['T_wkz_0']
Di = ['Durchmesser_innen']

setpoints_lab = ['Düsentemperatur', 'Werkzeugtemperatur', 
                 'Einspritzgeschwindigkeit', 'Umschaltpunkt', 'Nachdruckhöhe',
                 'Nachdruckzeit', 'Staudruck','Kühlzeit']

est_label = [label+'_est' for label in Di]
e_label = [label+'_error' for label in Di]


constraints = [('a_Temp_Model','>0'),
               ('a_Temp_Model','<1'),
               ('b_Temp_Model','>0'),
               ('b_Temp_Model','<1'),
               ('T1_Temp_Model','<0.5'),
               ('T2_Temp_Model','<1'),
               ('T1_Temp_Model','>1/10'),
               ('T2_Temp_Model','>1/10')]


# Dictionary for storing estimated models
Temp_Models = {}

# pandas dataframe as lookup table 
setpoints = pd.DataFrame(data=[], columns = setpoints_lab)

Charges = list(set((data_train['Charge'])))

for charge in Charges:
    
    # data_train_norm = Temp_Model.scale_data(data_train)
    # data_test_norm = Temp_Model.scale_data(data_test)
    
    data_charge = data_train.loc[data_train['Charge']==charge]

    setpoint = data_charge.iloc[[0]][setpoints_lab]
    setpoint.index = [charge]
    setpoints = pd.concat([setpoints,setpoint])
    
    
    Temp_Model = DoubleExponential(dim_u=1,dim_out=1,name='Temp_Model',
                               u_label=T, y_label = Di)

    data_charge = Temp_Model.scale_data(data_charge,scale_output=True)

    InitialParams = Temp_Model.data_initialization(data_charge)
    Temp_Model.Parameters = InitialParams

    res = ModelTraining(Temp_Model,data_charge,data_charge,
                    initializations=1,mode='static',
                    constraints=constraints)

    Temp_Model.Parameters = res.loc[0,'params_val']

    Temp_Models[charge] = Temp_Model
    

setpoints.index.name = 'Charge'

save = {'setpoints':setpoints,
        'Temp_Models':Temp_Models}

pkl.dump(save,open('Temp_Models.mdl','wb'))


# %% Plot some results 

sel_charges = np.random.choice(Charges,25)
sel_charges[4]=188

sel_charges = np.array([ 86, 243,   8, 142, 188,  46,  83, 100, 152, 140, 
                        248, 247, 174, 198, 149, 168, 125, 187, 172, 208, 
                        54,  76, 126,  60,  30])


fig,ax = plt.subplots(5,5)

i = 0

for c in sel_charges:
    model = Temp_Models[c]
    
    data_charge = data_train.loc[data_train['Charge']==c]
    data_charge = model.scale_data(data_charge,scale_output=True)
    
    data_plot = pd.DataFrame(data=np.arange(0,9,1/100),columns=['T_wkz_0'],
                             index=np.arange(0,9,1/100)) 
    
    _,pred = static_mode(model,data_plot)
     

    
    ax.flat[i].plot(data_plot['T_wkz_0'],
                           pred['Durchmesser_innen'],
                           linestyle='None', marker='x')    

    ax.flat[i].plot(data_charge['T_wkz_0'],
                           data_charge['Durchmesser_innen'],
                           linestyle='None', marker='o')                               
    i = i+1

# fig, ax = plt.subplots(5,5)
# ax.plot(data_charge.index.values,data_charge[Di].values)
# ax.plot(pred.index.values,pred[Di].values)








    


