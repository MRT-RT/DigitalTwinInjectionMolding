# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 13:24:25 2022

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

from DIM.miscellaneous.PreProcessing import LoadFeatureData, MinMaxScale
from functions import estimate_polynomial

plt.close('all')

path_data = path_dim / 'data/Versuchsplan/'

# Lade alle Daten


data_train,data_test  = LoadFeatureData(path_data.as_posix(),
                                        list(range(1,275)),'inner', True)

# Charges 131 and 144 have the same setpoint, put 144 of them into test dataset
charge_144 = data_train.loc[data_train['Charge']==144]

data_test = pd.concat([data_test,charge_144])

data_train = data_train.drop(index = charge_144.index)


columns = ['Düsentemperatur', 'Werkzeugtemperatur',
            'Einspritzgeschwindigkeit', 'Umschaltpunkt', 'Nachdruckhöhe',
            'Nachdruckzeit', 'Staudruck', 'Kühlzeit','T_wkz_0',
            'Durchmesser_innen']


# data_train_scale,scale = MinMaxScale(data_train,columns)
# data_test_scale,scale = MinMaxScale(data_test,columns)

# data_train_scale['Charge'] = data_train['Charge']
# data_test_scale['Charge'] = data_test['Charge']

save = {'data_train':data_train,
        'data_test':data_test}

pkl.dump(save,open('data_doubleExp.pkl','wb'))
