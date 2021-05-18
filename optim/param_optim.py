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
import pandas as pd
import pickle as pkl

from .DiscreteBoundedPSO import DiscreteBoundedPSO
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

def ControlInput(ref_trajectories,opti_vars,k):
    """
    Übersetzt durch Maschinenparameter parametrierte
    Führungsgrößenverläufe in optimierbare control inputs
    """
    
    control = []
            
    for key in ref_trajectories.keys():
        control.append(ref_trajectories[key](opti_vars,k))
    
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

def ModelTraining(model,data,initializations = 10, BFR=False, p_opts=None, s_opts=None):
    
   
    results = [] 
    
    for i in range(0,initializations):
        
        # in first run use initial model parameters (useful for online 
        # training when only time for one initialization) 
        if i > 0:
            model.Initialize()
        
        # Estimate Parameters on training data
        new_params = ModelParameterEstimation(model,data,p_opts,s_opts)
        
        # Assign new parameters to model
        model.Parameters = new_params
        
        # Evaluate on Validation data
        u_val = data['u_val']
        y_ref_val = data['y_val']
        init_state_val = data['init_state_val']

        # Loop over all experiments
        
        e = 0
        
        for j in range(0,u_val.shape[0]):   
               
            # Simulate Model
            y = model.Simulation(init_state_val[j],u_val[j])
            y = np.array(y)
                     
            e = e + cs.sumsqr(y_ref_val[j] - y) 
        
        # Calculate mean error over all validation batches
        e = e / u_val.shape[0]
        e = np.array(e).reshape((1,))
        
        # save parameters and performance in list
        results.append([e,new_params])
    
    results = pd.DataFrame(data = results, columns = ['loss','params'])
    
    return results 

def HyperParameterPSO(model,data,param_bounds,n_particles,options,
                      initializations=10,p_opts=None,s_opts=None):
    """
    Binary PSO for optimization of Hyper Parameters such as number of layers, 
    number of neurons in hidden layer, dimension of state, etc

    Parameters
    ----------
    model : model
        A model whose hyperparameters to be optimized are attributes of this
        object and whose model equations are implemented as a casadi function.
    data : dict
        A dictionary with training and validation data, see ModelTraining()
        for more information
    param_bounds : dict
        A dictionary with structure {'name_of_attribute': [lower_bound,upper_bound]}
    n_particles : int
        Number of particles to use
    options : dict
        options for the PSO, see documentation of toolbox.
    initializations : int, optional
        Number of times the nonlinear optimization problem is solved for 
        each particle. The default is 10.
    p_opts : dict, optional
        options to give to the optimizer, see Casadi documentation. The 
        default is None.
    s_opts : dict, optional
        options to give to the optimizer, see Casadi documentation. The 
        default is None.

    Returns
    -------
    hist, Pandas Dataframe
        Returns Pandas dataframe with the loss associated with each particle 
        in the first column and the corresponding hyperparameters in the 
        second column

    """
    
    path = 'temp/PSO_param/'
    
    # Formulate Particle Swarm Optimization Problem
    dimensions_discrete = len(param_bounds.keys())
    lb = []
    ub = []
    
    for param in param_bounds.keys():
        
        lb.append(param_bounds[param][0])
        ub.append(param_bounds[param][1])
    
    bounds= (lb,ub)
    
    # Define PSO Problem
    PSO_problem = DiscreteBoundedPSO(n_particles, dimensions_discrete, 
                                     options, bounds)

    # Make a directory and file for intermediate results 
    os.makedirs(path+model.name)

    for key in param_bounds.keys():
        param_bounds[key] = np.arange(param_bounds[key][0],
                                      param_bounds[key][1]+1,
                                      dtype = int)
    
    index = pd.MultiIndex.from_product(param_bounds.values(),
                                       names=param_bounds.keys())
    
    hist = pd.DataFrame(index = index, columns=['cost','model_params'])    
    
    pkl.dump(hist, open(path + model.name +'/' + 'HyperParamPSO_hist.pkl','wb'))
    
    # Define arguments to be passed to vost function
    cost_func_kwargs = {'model': model,
                        'param_bounds': param_bounds,
                        'n_particles': n_particles,
                        'dimensions_discrete': dimensions_discrete,
                        'initializations':initializations,
                        'p_opts': p_opts,
                        's_opts': s_opts,
                        'path':path}
    
    # Create Cost function
    def PSO_cost_function(swarm_position,**kwargs):
        
        # Load training history to avoid calculating stuff muliple times
        hist = pkl.load(open(path+ model.name +'/' +
                             'HyperParamPSO_hist.pkl','rb'))
            
       
        # Initialize empty array for costs
        cost = np.zeros((n_particles,1))
    
        for particle in range(0,n_particles):
            
            # Check if results for particle already exist in hist
            idx = tuple(swarm_position[particle].tolist())
            
            if (math.isnan(hist.loc[idx,'cost']) and
            math.isnan(hist.loc[idx,'model_params'])):
                
                # Adjust model parameters according to particle
                for p in range(0,dimensions_discrete):  
                    setattr(model,list(param_bounds.keys())[p],
                            swarm_position[particle,p])
                
                model.Initialize()
                
                # Estimate parameters
                results = ModelTraining(model,data,initializations, 
                                        BFR=False, p_opts=p_opts, 
                                        s_opts=s_opts)
                
                # Save all results of this particle in a file somewhere so that
                # the nonlinear optimization does not have to be done again
                
                pkl.dump(results, open(path + model.name +'/' + 'particle' + 
                                       str(swarm_position[particle]) + '.pkl',
                                       'wb'))
                
                # calculate best performance over all initializations
                cost[particle] = results.loss.min()
                
                # Save new data to dictionary for future iterations
                hist.loc[idx,'cost'] = cost[particle]
                
                # Save model parameters corresponding to best performance
                idx_min = pd.to_numeric(results['loss'].str[0]).argmin()
                hist.loc[idx,'model_params'] = \
                [results.loc[idx_min,'params']]
                
                # Save DataFrame to File
                pkl.dump(hist, open(path + model.name +'/' +
                                    'HyperParamPSO_hist.pkl','wb'))
                
            else:
                cost[particle] = hist.loc[idx].cost.item()
                
        
        
        
        cost = cost.reshape((n_particles,))
        return cost
    
    
    # Solve PSO Optimization Problem
    PSO_problem.optimize(PSO_cost_function, iters=100, **cost_func_kwargs)
    
    # Load intermediate results
    hist = pkl.load(open(path + model.name +'/' + 'HyperParamPSO_hist.pkl','rb'))
    
    # Delete file with intermediate results
    # os.remove(path + model.name +'/' + 'HyperParamPSO_hist.pkl')
    
    return hist

def ModelParameterEstimation(model,data,p_opts=None,s_opts=None):
    """
    

    Parameters
    ----------
    model : model
        A model whose hyperparameters to be optimized are attributes of this
        object and whose model equations are implemented as a casadi function.
    data : dict
        A dictionary with training and validation data, see ModelTraining()
        for more information
    p_opts : dict, optional
        options to give to the optimizer, see Casadi documentation. The 
        default is None.
    s_opts : dict, optional
        options to give to the optimizer, see Casadi documentation. The 
        default is None.

    Returns
    -------
    values : dict
        dictionary with either the optimal parameters or if the solver did not
        converge the last parameter estimate

    """
    
    
    u = data['u_train']
    y_ref = data['y_train']
    init_state = data['init_state_train']
    
    # Create Instance of the Optimization Problem
    opti = cs.Opti()
    
    params_opti = CreateOptimVariables(opti, model.Parameters)
    
    e = 0
    
    # Loop over all experiments/batches
    for i in range(0,u.shape[0]):   
           
        # Simulate Model
        y = model.Simulation(init_state[i],u[i],params_opti)
        
        e = e + cs.sumsqr(y_ref[i,:,:] - y)
    
    opti.minimize(e)
    
    # Create Solver
    
    # Solver options
    if p_opts is None:
        p_opts = {"expand":False}
    if s_opts is None:
        s_opts = {"max_iter": 1000, "print_level":0}
    
    opti.solver("ipopt",p_opts, s_opts)
    
    
    # Set initial values of Opti Variables as current Model Parameters
    for key in params_opti:
        opti.set_initial(params_opti[key], model.Parameters[key])
    
    
    # Solve NLP, if solver does not converge, use last solution from opti.debug
    try: 
        sol = opti.solve()
    except:
        sol = opti.debug
        
    values = OptimValues_to_dict(params_opti,sol)
    
    return values