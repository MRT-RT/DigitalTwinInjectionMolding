# -*- coding: utf-8 -*-

from sys import path
path.append(r"C:\Users\LocalAdmin\Documents\casadi-windows-py38-v3.5.5-64bit")

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np


def OptimValues_to_dict(optim_variables_dict,sol):
    # reads optimized parameters from optim solution and writes into dictionary
    
    values = {}
    
    for key in optim_variables_dict.keys():
       dim0 = optim_variables_dict[key].shape[0]
       dim1 = optim_variables_dict[key].shape[1]
       
       values[key] = sol.value(optim_variables_dict[key]) 
       
       # Convert tu numpy array
       values[key] = np.array(values[key]).reshape((dim0,dim1))

      
    return values

def RK4(f_cont,input,dt):
    '''
    Runge Kutta 4 numerical intergration method

    Parameters
    ----------
    f_cont : casadi function
        DESCRIPTION.
    dt : int
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    k1 = f_cont(*input)
    k2 = f_cont(*[input[0]+dt*k1/2,*input[1::]]) 
    k3 = f_cont(*[input[0]+dt*k2/2,*input[1::]])
    k4 = f_cont(*[input[0]+dt*k3,*input[1::]])
    
    x_new = input[0] + 1/6 * dt * (k1+2*k2+2*k3+k4)
    
    return x_new