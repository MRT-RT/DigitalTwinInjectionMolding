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



# path = '/home/alexander/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/normalized/'

path = 'E:/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/normalized/'
charges = list(range(1,275))
split = 'process'


# u_label_q = ['T_wkz_max', 't_Twkz_max', 'T_wkz_int', 'p_wkz_max',
#   'p_wkz_int', 'p_wkz_res', 't_pwkz_max']

u_label_p = ['Düsentemperatur', 'Werkzeugtemperatur', 'Einspritzgeschwindigkeit',
  'Umschaltpunkt', 'Nachdruckhöhe', 'Nachdruckzeit', 'Staudruck', 'Kühlzeit']
u_label_q = ['p_wkz_max', 't_pwkz_max', 't_Twkz_max']
y_label_q = ['Durchmesser_innen']

data_train,data_val = LoadFeatureData(path,charges,split,y_label_q)


# LoadSetpointData(path,charges,split,y_label_q)

# # Normalize data
data_train,minmax = MinMaxScale(data_train,u_label_q+u_label_p+y_label_q)
data_val,_ = MinMaxScale(data_val,u_label_q+u_label_p+y_label_q,minmax)


model_p = Static_MLP(dim_u=8, dim_out=3, dim_hidden=10,u_label=u_label_p,
                    y_label=u_label_q,name='proc', init_proc='xavier')

model_q = Static_MLP(dim_u=3, dim_out=1, dim_hidden=4,u_label=u_label_q,
                    y_label=y_label_q,name='qual', init_proc='xavier')


# result_p = ModelTraining(model_p,data_train,data_val,initializations=1,
#                           p_opts=None,s_opts=None,mode='static')

# result_q = ModelTraining(model_q,data_train,data_val,initializations=5,
#                           p_opts=None,s_opts=None,mode='static')

result_p = pkl.load(open('results_p_process.pkl','rb'))
result_q = pkl.load(open('results_q.pkl','rb'))



model_p.Parameters = result_p.loc[0]['params_val']
model_q.Parameters = result_q.loc[5]['params_val']

_,prediction_q = static_mode(model_q,data_val)
_,prediction_p = static_mode(model_p,data_val)


# print(BestFitRate(data_val[u_label_q].values, prediction_p[u_label_q].values))
print(BestFitRate(data_val[y_label_q].values, prediction_q[y_label_q].values))



fig, ax = plt.subplots(figsize=(20, 10))
sns.stripplot(x=data_val.index,y=data_val['p_wkz_max'],color='grey',alpha=.8,
              size=15,ax=ax)
sns.stripplot(x=prediction_p.index,y=prediction_p['p_wkz_max'],size=15,ax=ax)
ax.set_xlim([1,50]) 

# fig, ax = plt.subplots(figsize=(20, 10))
# sns.stripplot(x=data_val.index,y=data_val['Durchmesser_innen'],color='grey',alpha=.8,size=15,ax=ax)
# sns.stripplot(x=prediction.index,y=prediction['Durchmesser_innen'],size=15,ax=ax)
# ax.set_xlim([1,50]) 

# def Fit_MLP(dim_hidden):
    
#     # print(dim_hidden)
#     charges = list(range(1,275))
#     
    
#     split = 'all'
    
    
#     path = '/home/alexander/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
#     # path = 'E:/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
    
#     data_train,data_val,cycles_train_label,cycles_val_label,\
#         charge_train_label,charge_val_label = \
#             LoadStaticData(path,charges,split,targets)
    
#     # Normalize Data
#     data_max = data_train.max()
#     data_min = data_train.min()
    
#     data_train = 2*(data_train - data_min)/(data_max-data_min) - 1
#     data_val = 2*(data_val - data_min)/(data_max-data_min) - 1
    
#     inputs = [col for col in data_train.columns if col not in targets]
#     inputs = inputs[0:8]
    
#     data = {}
#     data['u_train'] = [data_train[inputs].values]
#     data['u_val'] = [data_val[inputs].values]
#     data['y_train'] = [data_train[targets].values]
#     data['y_val'] = [data_val[targets].values]


#     model = Static_MLP(dim_u=8, dim_out=1, dim_hidden=dim_hidden,name='MLP',
#                        init_proc='xavier')
    
#     s_opts = {"max_iter": 2000, 'hessian_approximation':'limited-memory'}
    


#     result['dim_hidden'] = dim_hidden
    
#     pkl.dump(result,open('MLP_Durchmesser_innen_dimhidden'+str(dim_hidden)+'.pkl','wb'))

#     return result



# if __name__ == '__main__':
    
#      h7 = Fit_MLP(dim_hidden=7)
#      h8 = Fit_MLP(dim_hidden=8)
#      h9 = Fit_MLP(dim_hidden=9)
#      h10 = Fit_MLP(dim_hidden=10)
    










