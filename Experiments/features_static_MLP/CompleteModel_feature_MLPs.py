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

u_label_q = ['T_wkz_0', 'T_wkz_max', 't_Twkz_max', 'T_wkz_int', 'p_wkz_max',
'p_wkz_int', 'p_wkz_res', 't_pwkz_max']                                         # = y_label_p

y_label_q = ['Durchmesser_innen']

data_train,data_val = LoadFeatureData(path,charges,split)


# LoadSetpointData(path,charges,split,y_label_q)

# # Normalize data
data_train,minmax = MinMaxScale(data_train,u_label_q+u_label_p+y_label_q)
data_val,_ = MinMaxScale(data_val,u_label_q+u_label_p+y_label_q,minmax)


model_p = Static_MLP(dim_u=8, dim_out=8, dim_hidden=40,u_label=u_label_p,
                    y_label=u_label_q,name='proc', init_proc='xavier')

model_q = Static_MLP(dim_u=8, dim_out=1, dim_hidden=4,u_label=u_label_q,
                    y_label=y_label_q,name='qual', init_proc='xavier')


# result_p = ModelTraining(model_p,data_train,data_val,initializations=5,
#                           p_opts=None,s_opts=None,mode='static')

# result_q = ModelTraining(model_q,data_train,data_val,initializations=5,
#                           p_opts=None,s_opts=None,mode='static')

result_p = pkl.load(open('results_p_process_stationary_40dim.pkl','rb'))
result_q = pkl.load(open('results_q_feature_stationary.pkl','rb'))

model_p.Parameters = result_p.loc[0]['params_val']
model_q.Parameters = result_q.loc[0]['params_val']

_,prediction_p = static_mode(model_p,data_val)
_,prediction_q = static_mode(model_q,prediction_p)


print(BestFitRate(data_val[y_label_q].values, prediction_q[y_label_q].values))



# fig, ax = plt.subplots(figsize=(20, 10))
# sns.stripplot(x=data_val.index,y=data_val[y_label_q[0]],color='grey',
#               alpha=.8,size=15,ax=ax)
# sns.stripplot(x=prediction_q.index,y=prediction_q[y_label_q[0]],
#               size=15,ax=ax)
# ax.set_xlim([1,50]) 










