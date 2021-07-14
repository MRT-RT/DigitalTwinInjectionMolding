# -*- coding: utf-8 -*-
from sys import path
path.append(r"C:\Users\LocalAdmin\Documents\casadi-windows-py38-v3.5.5-64bit")

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.model_structures import MLP
from optim.param_optim import HyperParameterPSO,ModelParameterEstimation,ModelTraining

# from miscellaneous import *

''' Generate dummy Identification Data for first phase '''
N = 100


u = np.zeros((10,N-1,2))
x = np.zeros((10,N,2))

for i in range(0,10):

    x_i = np.zeros((N,2))
    u_i = np.random.normal(0,1,(N-1,2))

    for k in range(1,100):
        x_i[k,0] = 0.7*x_i[k-1,0]**2 - 0.1*x_i[k-1,1]**2 + 0.1*u_i[k-1,0] 
        x_i[k,1] = 0.3*x_i[k-1,0]**2 + 0.3*x_i[k-1,1]**2 - 0.5*u_i[k-1,1]
    u[i,:,:] = u_i
    x[i,:,:] = x_i


init_state = x[:,0,:].reshape(10,2,1) 
data = {'u_train':u[0:8], 'y_train':x[0:8],'init_state_train': init_state[0:8],
        'u_val':u[8::], 'y_val':x[8::],'init_state_val': init_state[8::]}



''' Estimate MLP model for first phase '''
# Initialize Model
model = MLP(dim_u=2,dim_out=2,dim_hidden=10,name='Inject')
# y = model.Simulation(data['init_state_train'][0],data['u_train'][0])
# model = Model.LinearSSM(dim_u=2,dim_x=2,dim_y=2,name='Inject')
s_opts = {"max_iter": 1000, "print_level":0}
# y = model.Simulation(data['init_state_train'][0],data['u_train'][0])
# identified_parameters = ModelParameterEstimation(model,data,p_opts=None,s_opts=s_opts)

# identification_results = ModelTraining(model,data,initializations = 5)

# model.Parameters = identified_parameters

param_bounds = {'dim_hidden':np.array([5,20])}
options = {'c1': 0.6, 'c2': 0.3, 'w': 0.4, 'k':5, 'p':1}
n_particles = 5
initializations = 2
s_opts = {"max_iter": 100, "print_level":0}

results_inject = HyperParameterPSO(model,data,param_bounds,n_particles,
                            options,initializations,p_opts=None,s_opts=s_opts)


x0_train = data['init_state_train'][0]
u_train =  data['u_train'][0]
y_train =  data['y_train'][0]

y_est_train = model.Simulation(x0_train,u_train)
y_est_train = np.array(y_est_train)

plt.plot(y_train,label='True output')                                              # Plot True data
plt.plot(y_est_train,label='Est. output')                                          # Plot Model Output
plt.plot(y_train-y_est_train,label='Simulation Error')                             # Plot Error between model and true system (its almost zero)
plt.legend()
plt.show()




# model.dim_hidden = 6
# model.Initialize()
# model.Parameters = results_inject.loc[6,'model_params']

# model_inject = model

##############################################################################

''' Generate Identification Data for second phase '''
"""
N = 100

u = np.zeros((10,N-1,2))
x = np.zeros((10,N,2))

for i in range(0,10):

    x_i = np.zeros((N,2))
    u_i = np.random.normal(0,1,(N-1,2))

    for k in range(1,100):
        x_i[k,0] = 0.2*x_i[k-1,0]**2 + 0.2*x_i[k-1,1]**2 - 0.5*u_i[k-1,0] 
        x_i[k,1] = 0.6*x_i[k-1,0]**2 - 0.1*x_i[k-1,1]**2 + 0.1*u_i[k-1,1]
    u[i,:,:] = u_i
    x[i,:,:] = x_i


init_state = x[:,0,:].reshape(10,1,2) 
data = {'u_train':u[0:8], 'x_train':x[0:8],'init_state_train': init_state[0:8],
        'u_val':u[8::], 'x_val':x[8::],'init_state_val': init_state[8::]}



''' Estimate MLP model for first phase '''
model = Model.MLP(dim_u=2,dim_x=1,dim_hidden=10,name='Press')

param_bounds = {'dim_hidden':np.array([5,10])}
options = {'c1': 0.6, 'c2': 0.3, 'w': 0.4, 'k':5, 'p':1}
n_particles = 5
initializations = 5
s_opts = {"max_iter": 10, "print_level":0}

results_press = HyperParameterPSO(model,data,param_bounds,n_particles,
                            options,initializations,p_opts=None,s_opts=s_opts)



ProcessModel = Model.InjectionMouldingMachine()

PressurePhaseModel = Model.MLP(dim_u=2,dim_x=1,dim_hidden=8,name='PressurePhaseModel')
PressurePhaseModel.Parameters = results_press.loc[8,'model_params']

InjectionPhaseModel = Model.MLP(dim_u=2,dim_x=1,dim_hidden=8,name='InjectionPhaseModel')
InjectionPhaseModel.Parameters = results_inject.loc[8,'model_params']




ProcessModel.ModelInject = InjectionPhaseModel
ProcessModel.ModelPress = PressurePhaseModel


pkl.dump(ProcessModel,open('ProcessModel.pkl','wb'))

# model.dim_hidden = 6
# model.Initialize()
# model.Parameters = results.loc[6,'model_params']



# ''' Estimate Parameters RNN'''


# model = Model.GRU(dim_u=2,dim_c=1,dim_hidden=2,dim_out=2,name='GRU')

# data['init_state_train'] = np.zeros((8,1,1))
# data['init_state_val'] = np.zeros((2,1,1))
# data['x_train'] = x[0:8,-1,:].reshape(8,1,1)
# data['x_val'] = x[8::,-1,:].reshape(2,1,1)

# param_bounds = {'dim_c': np.array([1,5]), 'dim_hidden':np.array([2,10])}
# options = {'c1': 0.6, 'c2': 0.3, 'w': 0.4, 'k':5, 'p':1}
# n_particles = 5
# initializations = 5
# s_opts = {"max_iter": 10, "print_level":0}

# PSO_problem, hist = HyperParameterPSO(model,data,param_bounds,n_particles,
#                             options,initializations,p_opts=None,s_opts=s_opts)



''' Compare new and old model '''
  
# model.Parameters = new_params

# # Simulate Model
# x = model.Simulation(init_state[0,:,:],u_train[0,:,:])

# x = np.array(x)        


# np.linalg.norm(x_train[0,:,:]-x)
 


# plt.plot(x_train[0,-1,:])
# plt.plot(x)
"""