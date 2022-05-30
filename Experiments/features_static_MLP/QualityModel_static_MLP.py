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



# path = '/home/alexander/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/normalized/'

path = 'E:/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/normalized/'
charges = list(range(1,275))
split = 'all'

y_label = ['Durchmesser_innen']

u_label = ['Düsentemperatur', 'Werkzeugtemperatur', 'Einspritzgeschwindigkeit',
 'Umschaltpunkt', 'Nachdruckhöhe', 'Nachdruckzeit', 'Staudruck', 'Kühlzeit']

data_train,data_val = LoadSetpointData(path,charges,split)


# Normalize data
data_train,minmax = MinMaxScale(data_train,u_label+y_label)
data_val,_ = MinMaxScale(data_val,u_label+y_label,minmax)

model = Static_MLP(dim_u=8, dim_out=1, dim_hidden=5,u_label=u_label,
                   y_label=y_label,name='MLP', init_proc='xavier')

# p_opts = {"huhu": 1}

result = ModelTraining(model,data_train,data_val,initializations=1,p_opts=None,
                                s_opts=None,mode='static')

model.Parameters = result.loc[0,'params_val']

_,prediction = static_mode(model,data_val)

fig, ax = plt.subplots(figsize=(20, 10))
sns.stripplot(x=data_val.index,y=data_val['Durchmesser_innen'],color='grey',alpha=.8,size=15,ax=ax)
sns.stripplot(x=prediction.index,y=prediction['Durchmesser_innen'],size=15,ax=ax)
ax.set_xlim([1,50]) 

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
    










