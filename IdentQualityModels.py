# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:44:37 2021

@author: alexa
"""

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

from DIM.miscellaneous.PreProcessing import arrange_data_for_ident

from DIM.models.model_structures import GRU,LSTM
from DIM.models.injection_molding import QualityModel
from DIM.optim.param_optim import ModelTraining




dim_c = 2

# # Load cycle data, check if usable, convert to numpy array
cycles = []

for i in range(1,11):
    cycles.append(pkl.load(open('data/Versuchsplan/cycle'+str(i)+'.pkl','rb')))

cycles_train = cycles[3:8]
cycles_val = cycles[8:10]

# Select input and output for dynamic model
y_lab = ['Durchmesser_innen']
u_inj_lab= ['p_wkz_ist','T_wkz_ist','p_inj_ist','Q_Vol_ist','V_Screw_ist']
u_press_lab = u_inj_lab
u_cool_lab = ['p_wkz_ist','T_wkz_ist']
# 
x_train,q_train,switch_train  = arrange_data_for_ident(cycles_train,y_lab,
                                    u_inj_lab,u_press_lab,u_cool_lab,'quality')
#
# x_train,q_train,switch_train = arrange_data_for_qual_ident(cycles_train,x_lab,q_lab)

x_val,q_val,switch_val = arrange_data_for_ident(cycles_val,y_lab,
                                    u_inj_lab,u_press_lab,u_cool_lab,'quality')

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

injection_model = LSTM(dim_u=5,dim_c=dim_c,dim_hidden=5,dim_out=1,name='inject')
press_model = LSTM(dim_u=5,dim_c=dim_c,dim_hidden=5,dim_out=1,name='press')
cool_model = LSTM(dim_u=2,dim_c=dim_c,dim_hidden=5,dim_out=1,name='cool')

quality_model = QualityModel(subsystems=[injection_model,press_model,cool_model],
                              name='q_model')


# PSO gangbar machen!

results_LSTM = ModelTraining(quality_model,data,initializations=2, BFR=False, 
                  p_opts=None, s_opts=None)



# results = pkl.load(open('QualityModel_GRU_1c_5in_1out.pkl','rb'))

# quality_model.AssignParameters(results.loc[3,'params'])

# quality_model.switching_instances = data['switch_val'][0]

# c,y = quality_model.Simulation(data['init_state_val'][0], data['u_val'][0])


# plt.plot(data['u_val'][0])
# plt.plot(np.array(y))
# plt.plot(np.array(c))

# u_press_names = u_inj_names
# u_cool_names = u_inj_names

# # Choose measured process variables (is)
# # Keep temperatures out of the equation for now

# # Predict quality measurements : Gewicht, Durchmesser_innen

# inject,press,cool = arrange_data_for_ident(cycle1,x_names,u_inj_names,
#                                            u_press_names,u_cool_names)


# dim_c = 1




# quality_model.switching_instances =[14,339]

# u = cycle1['p_inj_ist'].values.reshape(1,-1,1)
# # c,y = quality_model.Simulation(np.zeros((10,1)),u)




# # values = ModelParameterEstimation(quality_model,data,p_opts=None,s_opts=None)

# results = ModelTraining(quality_model,data,initializations=10, BFR=False, 
#                   p_opts=None, s_opts=None)


# quality_model.AssignParameters(results.loc[9,'params'])

# c,y = quality_model.Simulation(data['init_state_train'][0],data['u_train'][0])

# plt.plot(y)
# plt.plot(c)


















