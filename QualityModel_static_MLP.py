# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 15:16:22 2022

@author: LocalAdmin
"""


from DIM.miscellaneous.PreProcessing import LoadStaticData,LoadDynamicData
from DIM.models.model_structures import Static_MLP
from DIM.optim.param_optim import ModelParameterEstimation


charges = list(range(1,275))
targets = ['Durchmesser_innen']

path = './data/Versuchsplan/'

data_train,data_val = LoadStaticData(path,charges,targets)

inputs = [col for col in data_train.columns if col not in targets]
inputs = inputs[0:8]

data = {}
data['u_train'] = [data_train[inputs].values]
data['u_val'] = [data_val[inputs].values]
data['y_train'] = [data_train[targets].values]
data['y_val'] = [data_val[targets].values]


model = Static_MLP(dim_u=8, dim_out=1, dim_hidden=11,name='MLP')
result = ModelParameterEstimation(model,data,p_opts=None,s_opts=None,mode='static')



# dim_c = 2

# u_inj_lab= ['p_wkz_ist','T_wkz_ist','p_inj_ist','Q_Vol_ist','V_Screw_ist']
# u_press_lab = ['p_wkz_ist','T_wkz_ist','p_inj_ist','Q_Vol_ist','V_Screw_ist']
# u_cool_lab = ['p_wkz_ist','T_wkz_ist','p_inj_ist','Q_Vol_ist','V_Screw_ist']

# u_lab = [u_inj_lab]

# y_lab = ['Durchmesser_innen']

# data,cycles_train_label,cycles_val_label,charge_train_label,charge_val_label = \
# LoadDynamicData(path,charges,y_lab,u_lab)













