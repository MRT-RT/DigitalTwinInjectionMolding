# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 15:16:22 2022

@author: LocalAdmin
"""
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "/home/alexander/GitHub/DigitalTwinInjectionMolding/")
sys.path.insert(0, 'E:/GitHub/DigitalTwinInjectionMolding/')

from DIM.miscellaneous.PreProcessing import LoadFeatureData,LoadSetpointData,MinMaxScale
from DIM.models.model_structures import Static_MLP
from DIM.optim.param_optim import ModelTraining, static_mode
from DIM.optim.common import BestFitRate



path = '/home/alexander/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/normalized/'

# path = 'E:/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/normalized/'
charges = list(range(1,275))
split = 'process'

u_label_p = ['Düsentemperatur', 'Werkzeugtemperatur', 'Einspritzgeschwindigkeit',
  'Umschaltpunkt', 'Nachdruckhöhe', 'Nachdruckzeit', 'Staudruck', 'Kühlzeit']
y_label_p = ['T_wkz_0', 'T_wkz_max', 't_Twkz_max', 'T_wkz_int', 'p_wkz_max',
'p_wkz_int', 'p_wkz_res', 't_pwkz_max']

data_train,data_val = LoadFeatureData(path,charges,split)

# LoadSetpointData(path,charges,split,y_label_q)

# # Normalize data
data_train,minmax = MinMaxScale(data_train,u_label_p+y_label_p)
data_val,_ = MinMaxScale(data_val,u_label_p+y_label_p,minmax)

model_p = Static_MLP(dim_u=8, dim_out=8, dim_hidden=40,u_label=u_label_p,
                    y_label=y_label_p,name='proc', init_proc='xavier')

# result_p = ModelTraining(model_p,data_train,data_val,initializations=5,
#                           p_opts=None,s_opts=None,mode='static')

result_p = pkl.load(open('results_p_process_stationary_40dim.pkl','rb'))
# pkl.dump(result_p,open('results_p_process_stationary.pkl','wb'))

model_p.Parameters = result_p.loc[4]['params_val']
# model_q.Parameters = result_q.loc[0]['params_val']

_,prediction_p = static_mode(model_p,data_val)

print(BestFitRate(data_val[y_label_p].values, prediction_p[y_label_p].values))

# fig, ax = plt.subplots(figsize=(20, 10))
# sns.stripplot(x=data_val.index,y=data_val['p_wkz_max'],color='grey',alpha=.8,
#               size=15,ax=ax)
# sns.stripplot(x=prediction_p.index,y=prediction_p['p_wkz_max'],size=15,ax=ax)
# ax.set_xlim([1,50]) 
