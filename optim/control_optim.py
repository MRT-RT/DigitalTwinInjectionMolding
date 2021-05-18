#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 13:25:16 2020

@author: alexander
"""

from sys import path
path.append(r"C:\Users\LocalAdmin\Documents\casadi-windows-py38-v3.5.5-64bit")

import os

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import math
# from DiscreteBoundedPSO import DiscreteBoundedPSO
import pandas as pd
import pickle as pkl

from .common import OptimValues_to_dict

# Import sphere function as objective function
#from pyswarms.utils.functions.single_obj import sphere as f

# Import backend modules
# import pyswarms.backend as P
# from pyswarms.backend.topology import Star
# from pyswarms.discrete.binary import BinaryPSO

# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython


# from miscellaneous import *


# def SimulateModel(model,x,u,params=None):
#     # Casadi Function needs list of parameters as input
#     if params==None:
#         params = model.Parameters
    
#     params_new = []
        
#     for name in  model.Function.name_in():
#         try:
#             params_new.append(params[name])                      # Parameters are already in the right order as expected by Casadi Function
#         except:
#             continue
    
#     x_new = model.Function(x,u,*params_new)     
                          
#     return x_new

def ControlInput(reference,opti_vars,k):
    """
    Übersetzt durch Maschinenparameter parametrierte
    Führungsgrößenverläufe in optimierbare control inputs
    """
    
    if reference == []:
        return []
            
    control = []
    
    for ref in reference:
        control.append(ref(opti_vars,k))
    
    control = cs.vcat(control)

    return control   
    
def CreateOptimVariables(opti, RefTrajectoryParams):
    """
    Defines all parameters, which parameterize reference trajectories, as
    opti variables and puts them in a large dictionary
    """
    
    # Create empty dictionary
    opti_vars = {}
    
    for param in RefTrajectoryParams.keys():
        
        dim0 = RefTrajectoryParams[param].shape[0]
        dim1 = RefTrajectoryParams[param].shape[1]
        
        opti_vars[param] = opti.variable(dim0,dim1)
    
    # Create one parameter dictionary for each phase
    # opti_vars['RefParamsInject'] = {}
    # opti_vars['RefParamsPress'] = {}
    # opti_vars['RefParamsCool'] = {}

    # for key in opti_vars.keys():
        
    #     param_dict = getattr(process_model,key)
        
    #     if param_dict is not None:
        
    #         for param in param_dict.keys():
                
    #             dim0 = param_dict[param].shape[0]
    #             dim1 = param_dict[param].shape[1]
                
    #             opti_vars[key][param] = opti.variable(dim0,dim1)
    #     else:
    #         opti_vars[key] = None
  
    return opti_vars

def MultiStageOptimization(process_model,target):
    
    subsystems = process_model.subsystems
    switching_instances = process_model.switching_instances
    reference = process_model.reference
    ref_params = process_model.ref_params
    
    output_dims = [sys.dim_out for sys in subsystems]
    
    # Plausibility and Dimension Checks
    if len(subsystems) != len(reference):
        print('Number of Subsystems does not equal number of reference signals to be optmized!')
    elif len(switching_instances) != len(subsystems)-1:
        print('Number of switching instances does not fit number of Subsystems!')
    elif output_dims.count(output_dims[0]) != len(subsystems)*output_dims[0]:
        print('All Subsystems need to have the same output dimension')        
    
        
    # Create Instance of the Optimization Problem
    opti = cs.Opti()
    
    # Translate Maschinenparameter into opti.variables
    ref_params_opti = CreateOptimVariables(opti, ref_params)
    
    # Number of time steps
    N = target.shape[0]
    
    # Create decision variables for states
    X = opti.variable(N,output_dims[0])
        
    # Initial Constraints
    opti.subject_to(X[0]==target[0])
    
    # Generate an index vector pointing to the active subsystem in each time step
    active_subsystem = np.zeros(N,np.int8)

    for switch in switching_instances:
        active_subsystem[switch::] = active_subsystem[switch::]+1
    
    # System Dynamics as Path Constraints
    for k in range(N-1):
        
        # Control input at every time step is a function of the parameters of
        # the reference trajectories 
        U = ControlInput(reference[active_subsystem[k]],ref_params_opti,k)
        
        # Do a one step prediction based on the model
        pred = subsystems[active_subsystem[k]].OneStepPrediction(X[k],U)
        
        # OneStepPrediction can return a tuple, i.e. state and output. The 
        # output is per convention the second entry
        if isinstance(pred,tuple):
            pred = pred[1]
        
        opti.subject_to(pred==X[k+1])
            
    ''' Further Path Constraints (to avoid values that might damage the 
    machine or are in other ways harmful or unrealistic) '''
    
    # TO DO #
    
    
    ''' Final constraint might make solution infeasible, search for methods
    of relaxation''' 
    # opti.subject_to(X[-1]==target[-1])
    
    
    # Set initial values for Machine Parameters
    for key in ref_params_opti:
        opti.set_initial(ref_params_opti[key],ref_params[key])

    # Set initial values for state trajectory ??
    # for key in model.Maschinenparameter_opti:
    #     opti.set_initial(model.Maschinenparameter_opti[key],CurrentParams[key])      
    
    # Define Loss Function    
    opti.minimize(cs.sumsqr(X-target))
    
    #Choose solver
    opti.solver('ipopt')
    
    # Get solution
    sol = opti.solve()
    
    # Extract real values from solution
    values = OptimValues_to_dict(ref_params_opti,sol)
    values['X'] = sol.value(X)

    
    return values

def SingleStageOptimization(model,ref,N):
    """ 
    single shooting procedure for optimal control of a scalar final value
    
    model: Quality Model
    ref: skalarer Referenzwert für Optimierungsproblem
    N: Anzahl an Zeitschritten
    """
    
    # Create Instance of the Optimization Problem
    opti = cs.Opti()
    
    # Create decision variables for states
    U = opti.variable(N,1)
        
    # Initial quality 
    x = 0
    y = 0
    X = [x]
    Y = [y]
    
    # Simulate Model
    for k in range(N):
        out = SimulateModel(model.ModelQuality,X[k],U[k],model.ModelParamsQuality)
        X.append(out[0])
        Y.append(out[1])
            
    X = hcat(X)
    Y = hcat(Y)
    
    # Define Loss Function  
    opti.minimize(sumsqr(Y[-1]-ref))
                  
    #Choose solver
    opti.solver('ipopt')
    
    # Get solution
    sol = opti.solve()   

    # Extract real values from solution
    values = {}
    values['U'] = sol.value(U)
    
    return values






