# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:44:37 2021

@author: alexa
"""

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

from DIM.miscellaneous.PreProcessing import arrange_data_for_ident, eliminate_outliers

from DIM.models.model_structures import GRU,LSTM
from DIM.models.injection_molding import QualityModel
from DIM.optim.param_optim import ModelTraining, HyperParameterPSO



dim_c = 2


# Load Versuchsplan to find cycles that should be considered for modelling
data = pkl.load(open('./data/Versuchsplan/Versuchsplan.pkl','rb'))

data = eliminate_outliers(data)

# Delete outliers rudimentary

# Cycles for parameter estimation
cycles_train_label = []
cycles_val_label = []

for charge in range(1,5):
    cycles_train_label.append(data[data['Charge']==charge].index.values[-3:-1])
    cycles_val_label.append(data[data['Charge']==charge].index.values[-1])

cycles_train_label = list(cycles_train_label)
cycles_val_label = list(cycles_val_label)


# ''' FOR DEBUGGING PURPOSES'''
# cycles_train_label = cycles_train_label[0:10]

# Delete cycles that for some reason don't exist
cycles_train_label = list(np.delete(cycles_train_label, np.where(cycles_train_label == 767)))



# # Load cycle data, check if usable, convert to numpy array
cycles_train = []
cycles_val = []

for c in cycles_train_label:
    cycles_train.append(pkl.load(open('data/Versuchsplan/cycle'+str(c)+'.pkl',
                                      'rb')))

for c in cycles_val_label:
    cycles_val.append(pkl.load(open('data/Versuchsplan/cycle'+str(c)+'.pkl',
                                      'rb')))
    
    
# Select input and output for dynamic model
y_lab = ['Durchmesser_innen']
u_inj_lab= ['p_wkz_ist','T_wkz_ist'] #['p_wkz_ist','T_wkz_ist','p_inj_ist','Q_Vol_ist','V_Screw_ist']
u_press_lab = ['p_wkz_ist','T_wkz_ist'] #u_inj_lab
u_cool_lab = ['p_wkz_ist','T_wkz_ist']
# 

# Normalize with respect to first cycle    
mean_u = cycles_train[0][u_inj_lab].mean()
mean_y = cycles_train[0][y_lab].mean()
min_u = cycles_train[0][u_inj_lab].min()
max_u = cycles_train[0][u_inj_lab].max()


for cycle in cycles_train+cycles_val:
    cycle[u_inj_lab] = (cycle[u_inj_lab]-min_u)/(max_u-min_u)
    cycle[y_lab] = cycle[y_lab]-mean_y

#
x_train,q_train,switch_train  = arrange_data_for_ident(cycles_train,y_lab,
                                    [u_inj_lab,u_press_lab,u_cool_lab],'quality')
#
# x_train,q_train,switch_train = arrange_data_for_qual_ident(cycles_train,x_lab,q_lab)

x_val,q_val,switch_val = arrange_data_for_ident(cycles_val,y_lab,
                                    [u_inj_lab,u_press_lab,u_cool_lab],'quality')

c0_train = [np.zeros((dim_c,1)) for i in range(0,len(x_train))]
c0_val = [np.zeros((dim_c,1)) for i in range(0,len(x_val))]

data = {'u_train': x_train,
        'y_train': q_train,
        'switch_train': switch_train,
        'init_state_train': c0_train,
        'u_val': x_val,
        'y_val': q_val,
        'switch_val': switch_val,
        'init_state_val': c0_val}

#
injection_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=10,dim_out=1,name='inject')
press_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=10,dim_out=1,name='press')
cool_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=10,dim_out=1,name='cool')

quality_model = QualityModel(subsystems=[injection_model,press_model,cool_model],
                              name='q_model_Durchmesser_innen')


# param_bounds = {'dim_c':[1,10],'dim_hidden':[1,10]} 

# options = {'c1': 0.6, 'c2': 0.3, 'w': 0.4, 'k':5, 'p':1}


s_opts = {"hessian_approximation": 'limited-memory',"max_iter": 2000,
          "print_level":2}


# hist =  HyperParameterPSO(quality_model,data,param_bounds,n_particles=20,
#                           options = options, initializations=15,p_opts=None,
#                           s_opts=s_opts)

results_GRU = ModelTraining(quality_model,data,initializations=10, BFR=False, 
                  p_opts=None, s_opts=s_opts)

pkl.dump(results_GRU,open('GRU_'+str(*y_lab)+'_test.pkl','wb'))


# Assign and evaluate
quality_model.AssignParameters(results_GRU.loc[0]['params'])

x,y = quality_model.Simulation(c0=c0_val[0], u=x_val[0])


plt.plot(np.array(x))