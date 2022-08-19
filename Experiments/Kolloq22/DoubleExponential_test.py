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
               ('T1_Temp_Model','<40'),
               ('T2_Temp_Model','<40'),
               ('T1_Temp_Model','>0'),
               ('T2_Temp_Model','>0')]


# Dictionary for storing estimated models
Temp_Models = {}

# pandas dataframe as lookup table 
setpoints = pd.DataFrame(data=[], columns = setpoints_lab)

for charge in list(set((data_train['Charge']))):
    
    # data_train_norm = Temp_Model.scale_data(data_train)
    # data_test_norm = Temp_Model.scale_data(data_test)
    
    data_charge = data_train.loc[data_train['Charge']==charge]

    setpoint = data_charge.iloc[[0]][setpoints_lab]
    setpoint.index = [charge]
    setpoints = pd.concat([setpoints,setpoint])
    
    
    Temp_Model = DoubleExponential(dim_u=1,dim_out=1,name='Temp_Model',
                               u_label=T, y_label = Di)

    data_charge = Temp_Model.scale_data(data_charge)

    InitialParams = Temp_Model.data_initialization(data_charge)
    Temp_Model.Parameters = InitialParams

    res = ModelTraining(Temp_Model,data_charge,data_charge,
                    initializations=1,mode='static',
                    constraints=constraints)

    Temp_Model.Parameters = res.loc[0,'params_val']

    Temp_Models[charge] = Temp_Model
    

save = {'setpoints':setpoints,
        'Temp_Models':Temp_Models}

pkl.dump(save,open('Temp_Models.mdl','wb'))

# _,pred = static_mode(Temp_Models[charge_],data_charge)


# fig, ax = plt.subplots(1,1)
# ax.plot(data_charge.index.values,data_charge[Di].values)
# ax.plot(pred.index.values,pred[Di].values)








    


