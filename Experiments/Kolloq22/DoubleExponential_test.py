# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 10:38:18 2022

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
from DIM.models.model_structures import DoubleExponential
from DIM.optim.param_optim import ModelTraining, static_mode

# plt.close('all')

# path_data = path_dim / 'data/Versuchsplan/'

# Lade alle Daten
cross_val = False
normalize = True

charge = 25

data = pkl.load(open('data_minmax.pkl','rb'))

data_train = data['data_train']
data_test = data['data_test']

data_train=data_train.loc[data_train['Charge']==charge]
data_test=data_test.loc[data_test['Charge']==charge]

data_all = pd.concat([data_train,data_test])

inputs = ['T_wkz_0']
targets = ['Durchmesser_innen']

est_label = [label+'_est' for label in targets]
e_label = [label+'_error' for label in targets]

Temp_Model = DoubleExponential(dim_u=1,dim_out=1,name='Temp_Model',
                               u_label=inputs, y_label = targets)

# b=1

# c = Temp_Model.test(b)

# data_train_norm = Temp_Model.scale_data(data_train)
# data_test_norm = Temp_Model.scale_data(data_test)

# data_all_norm = pd.concat([data_train_norm,data_test_norm])

data_train_norm = Temp_Model.scale_data(data_all)
data_test_norm = data_train_norm

Temp_Model.data_initializtion(data_train_norm)


constraints = [('a_Temp_Model','>0'),
               ('a_Temp_Model','<1'),
               ('b_Temp_Model','>0'),
               ('b_Temp_Model','<1'),
               ('T1_Temp_Model','<40'),
               ('T2_Temp_Model','<40'),
               ('T1_Temp_Model','>0'),
               ('T2_Temp_Model','>0')]

res = ModelTraining(Temp_Model,data_train_norm,data_test_norm,
                    initializations=1,mode='static',constraints=constraints)


Temp_Model.Parameters = res.loc[0,'params_val']

print(Temp_Model.Parameters )

data_test_norm = pd.DataFrame(data = np.arange(0,0.4,1/1000),columns=inputs)

_,y_est = static_mode(Temp_Model,data_test_norm)

fig,ax = plt.subplots(1,1)

ax.plot(data_train_norm[inputs].values,data_train_norm[targets].values,
        linestyle = 'None', marker='o')
ax.plot(data_test_norm[inputs].values,y_est[targets].values,
        linestyle = 'None', marker='x')
