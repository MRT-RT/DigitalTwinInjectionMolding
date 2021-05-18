# -*- coding: utf-8 -*-
from sys import path
path.append(r"C:\Users\LocalAdmin\Documents\casadi-windows-py38-v3.5.5-64bit")

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import modellklassen as Model
from OptimizationTools import *
from miscellaneous import *

''' Generate Identification Data from an arbitrary state space model with 
two states and one output (it's just data to check if the code works) '''

N = 100

u = np.zeros((10,N-1,2))
x = np.zeros((10,N,2))
y = np.zeros((10,N-1,2))


for i in range(0,10):
    
    u_i = np.random.normal(0,1,(N-1,2))
    
    x_i = np.zeros((N,2))
    y_i = np.zeros((N-1,2))
    
    for k in range(1,100):
        # x_i[k,0] = 0.7*x_i[k-1,0]**2 - 0.1*x_i[k-1,1]**2 + 0.1*u_i[k-1,0] 
        # x_i[k,1] = 0.3*x_i[k-1,0]**2 + 0.3*x_i[k-1,1]**2 - 0.5*u_i[k-1,1]
        x_i[k,0] = 0.7*x_i[k-1,0] - 0.1*x_i[k-1,1] + 1*u_i[k-1,0] 
        x_i[k,1] = 0.3*x_i[k-1,0] + 0.3*x_i[k-1,1] - 2*u_i[k-1,1]
        y_i[k-1,0] = x_i[k,0]
        y_i[k-1,1] = x_i[k,1]
    
    u[i,:,:] = u_i
    x[i,:,:] = x_i
    y[i,:,:] = y_i

init_state = x[:,0,:].reshape(10,2,1) 

# Arrange Training and Validation data in a dictionary with the following
# structure. Thes dictionary must have these keys
data = {'u_train':u[0:8], 'y_train':y[0:8],'init_state_train': init_state[0:8],
        'u_val':u[8::], 'y_val':y[8::],'init_state_val': init_state[8::]}


''' Define the model as a linear state space model with name 'test' '''
model = Model.LinearSSM(dim_u=2,dim_x=2,dim_y=2,name='test')



''' Call the Function ModelTraining, which takes the model and the data and 
starts the optimization procedure 'initializations'-times. '''
identification_results = ModelTraining(model,data,initializations = 10)


''' The output is a pandas dataframe which contains the results for each of
the 10 initializations, specifically the loss on the validation data
and the estimated parameters ''' 

# Pick the parameters from the second initialization (for example, in this case
# every model has a loss close to zero because the optimizer is really good
# and its 'only' a linear model which we identify)

model.Parameters = identification_results.loc[2,'params']


# Maybe plot the simulation result to see how good the model performs
y_est = model.Simulation(init_state[0],u[0])
y_est = np.array(y_est)  
plt.plot(y[0],label='True output')                                                    # Plot True data
plt.plot(y_est,label='Est. output')                                             # Plot Model Output
plt.plot(y[0]-y_est,label='Simulation Error')                                   # Plot Error between model and true system (its almost zero)
plt.legend()
plt.show()
