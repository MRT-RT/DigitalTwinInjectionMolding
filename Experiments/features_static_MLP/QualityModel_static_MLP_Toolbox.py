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

from DIM.miscellaneous.PreProcessing import LoadSetpointData,MinMaxScale
from DIM.models.model_structures import Static_MLP
from DIM.optim.param_optim import ModelTraining, static_mode
from DIM.optim.common import BestFitRate


path = '/home/alexander/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/normalized/'

# path = 'E:/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/normalized/'
charges = list(range(1,275))
split = 'all'


u_label_q = ['Düsentemperatur', 'Werkzeugtemperatur', 'Einspritzgeschwindigkeit',
 'Umschaltpunkt', 'Nachdruckhöhe', 'Nachdruckzeit', 'Staudruck', 'Kühlzeit']

y_label_q = ['Durchmesser_innen']

data_train,data_val = LoadSetpointData(path,charges,split)

# Normalize data
data_train,minmax = MinMaxScale(data_train,u_label_q+y_label_q)
data_val,_ = MinMaxScale(data_val,u_label_q+y_label_q,minmax)

model_q = Static_MLP(dim_u=8, dim_out=1, dim_hidden=10,u_label=u_label_q,
                   y_label=y_label_q,name='qual', init_proc='xavier')

result_q = ModelTraining(model_q,data_train,data_val,initializations=5,p_opts=None,
                                s_opts=None,mode='static')

# pkl.dump(result_q,open('results_q_static_stationary.pkl','wb'))
result_q = pkl.load(open('results_q_static_trans.pkl','rb'))


model_q.Parameters = result_q.loc[0]['params_val']


_,prediction_q = static_mode(model_q,data_val)


print(BestFitRate(data_val[y_label_q].values, prediction_q[y_label_q].values))

# fig, ax = plt.subplots(figsize=(20, 10))
# sns.stripplot(x=data_val.index,y=data_val[y_label_q[0]],color='grey',
#               alpha=.8,size=15,ax=ax)
# sns.stripplot(x=prediction_q.index,y=prediction_q[y_label_q[0]],
#               size=15,ax=ax)
# ax.set_xlim([1,50]) 

