#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 13:25:16 2020

@author: alexander
"""

from sys import path
path.append(r"C:\Users\LocalAdmin\Documents\casadi-windows-py38-v3.5.5-64bit")

import os
import time
import copy

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import pickle as pkl


from DIM.optim.DiscreteBoundedPSO import DiscreteBoundedPSO
from .common import OptimValues_to_dict,BestFitRate

import multiprocessing


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
    
# def CreateOptimVariables(opti, Parameters):
#     '''
#     Beschreibung der Funktion

#     Parameters
#     ----------
#     opti : Dict
#         DESCRIPTION.
#     Parameters : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     opti_vars : TYPE
#         DESCRIPTION.

#     '''
        
#     # Create empty dictionary
#     opti_vars = {}
    
#     for param in Parameters.keys():
#         dim0 = Parameters[param].shape[0]
#         dim1 = Parameters[param].shape[1]
        
#         opti_vars[param] = opti.variable(dim0,dim1)
    
#     return opti_vars

def CreateOptimVariables(Parameters):
    '''
    Beschreibung der Funktion

    Parameters
    ----------
    opti : Dict
        DESCRIPTION.
    Parameters : TYPE
        DESCRIPTION.

    Returns
    -------
    opti_vars : TYPE
        DESCRIPTION.

    '''
        
    # Create empty dictionary
    opti_vars = {}
    opti_vars_vec = []
    for param in Parameters.keys():
        dim0 = Parameters[param].shape[0]
        dim1 = Parameters[param].shape[1]
        
        opti_vars[param] = cs.MX.sym(param,dim0,dim1)
        opti_vars_vec.append(opti_vars[param].reshape((-1,1)))
        
    return opti_vars, cs.vcat(opti_vars_vec)


def ModelTraining(model,data,initializations=10, BFR=False, 
                  p_opts=None, s_opts=None,mode='parallel'):
    
   
    results = [] 
    
    for i in range(0,initializations):
        
        # initialize model to make sure given initial parameters are assigned
        model.ParameterInitialization()
        
        # Estimate Parameters on training data
        new_params,loss_train,loss_val = ModelParameterEstimation(model,data,p_opts,s_opts,mode)
       
        # save parameters and performance in list
        results.append([loss_train,loss_val,model.name,model.Parameters])
           
    results = pd.DataFrame(data = results, columns = ['loss_train','loss_val',
                        'model','params'])
    return results

def TrainingProcedure(model, data, p_opts, s_opts, mode):
    
    # initialize model to make sure given initial parameters are assigned
    model.ParameterInitialization()
    
    # Estimate Parameters on training data
    new_params,loss_train,loss_val = ModelParameterEstimation(model,data,p_opts,s_opts,mode)
    
    # save parameters and performance in list
    result = [loss_train,loss_val,model.name,model.Parameters]
    
    return result

def ParallelModelTraining(model,data,initializations=10, BFR=False, 
                  p_opts=None, s_opts=None,mode='parallel'):
    
     
    data = [copy.deepcopy(data) for i in range(0,initializations)]
    model = [copy.deepcopy(model) for i in range(0,initializations)]
    p_opts = [copy.deepcopy(p_opts) for i in range(0,initializations)]
    s_opts = [copy.deepcopy(s_opts) for i in range(0,initializations)]
    mode = [copy.deepcopy(mode) for i in range(0,initializations)]
    
    pool = multiprocessing.Pool(10)
    results = pool.starmap(TrainingProcedure, zip(model, data, p_opts, s_opts, mode))        
    results = pd.DataFrame(data = results, columns = ['loss_train','loss_val',
                        'model','params'])
    
    pool.close() 
    pool.join()      
    
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
    try:
        os.makedirs(path+model.name)
    
        for key in param_bounds.keys():
            param_bounds[key] = np.arange(param_bounds[key][0],
                                          param_bounds[key][1]+1,
                                          dtype = int)
        
        index = pd.MultiIndex.from_product(param_bounds.values(),
                                           names=param_bounds.keys())
        
        hist = pd.DataFrame(index = index, columns=['cost','model_params'])    
        
        pkl.dump(hist, open(path + model.name +'/' + 'HyperParamPSO_hist.pkl','wb'))
    
    except:
        print('Found data, PSO continues...')
        
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
                cost[particle] = results.loss_val.min()
                
                # Save new data to dictionary for future iterations
                hist.loc[idx,'cost'] = cost[particle]
                
                # Save model parameters corresponding to best performance
                idx_min = results['loss_val'].idxmin()
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
    PSO_problem.optimize(PSO_cost_function, iters=100, n_processes=None,**cost_func_kwargs)
    
    # Load intermediate results
    hist = pkl.load(open(path + model.name +'/' + 'HyperParamPSO_hist.pkl','rb'))
    
    # Delete file with intermediate results
    # os.remove(path + model.name +'/' + 'HyperParamPSO_hist.pkl')
    
    return hist

def ModelParameterEstimation(model,data,p_opts=None,s_opts=None,mode='parallel'):
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
    
    max_iter = s_opts['max_iter']
    
    # Create dictionary of all non-frozen parameters to create Opti Variables of 
    OptiParameters = model.Parameters.copy()
   
    for frozen_param in model.FrozenParameters:
        OptiParameters.pop(frozen_param)
        
    
    params_opti,opti_vars_vec = CreateOptimVariables(OptiParameters)
    
    
    # Evaluate on model on data
    u_train = data['u_train']
    y_ref_train = data['y_train']

    u_val = data['u_val']
    y_ref_val = data['y_val']
    
    try:
        switch_train =  data['switch_train']
        switch_val =  data['switch_val']
    except KeyError:
        switch = None
    
    if mode == 'parallel':
        x0_train = data['init_state_train']
        x0_val = data['init_state_val']
        
        loss_train,_,_,_ = parallel_mode(model,u_train,y_ref_train,x0_train,
                                         switch_train,params_opti) 
        
        loss_val,_,_,_ = parallel_mode(model,u_val,y_ref_val,x0_val,
                                       switch_val,params_opti)
        
    elif mode == 'static':
        loss_train,_,_ = static_mode(model,u,y_ref,params_opti)   
    elif mode == 'series':
        x0 = data['init_state_train']
        loss_train,_,_ = series_parallel_mode(model,u,y_ref,x0,params_opti)
    
    
    # LM Optimizer
   
    grad = cs.gradient(loss_train,opti_vars_vec)
    hess = cs.mtimes(grad,grad.T)
    nlp = cs.Function('loss_train',[*list(params_opti.values())],
                         [loss_train,grad,hess])
    nlp_val = cs.Function('loss_val',[*list(params_opti.values())],
                         [loss_val])
        
    theta = model.Parameters.copy()
    
    lam = 1
    step = 0.01
    
    nlp_val_hist = np.inf
    
    for i in range(0,max_iter):

        nlp_train,nlp_grad,nlp_hess = nlp(*list(theta.values()))
        nlp_v = nlp_val(*list(theta.values()))
        
        print('Iteration: '+str(i) + '   loss_train: ' + str(nlp_train) + \
              '   loss_val: ' + str(nlp_v))
            
        F = nlp_train
        G = nlp_grad
        H = nlp_hess  
        
        improvement = False
        
        while improvement is False:
            
            d_theta = -step*cs.mtimes(cs.inv(H+lam*np.eye(H.shape[0])),G)*F
            
            # new params
            theta_new =  AddParameterUpdate(theta,d_theta)
            
            # evaluate loss
            nlp_f,_,_ = nlp(*list(theta_new.values()))
            
            if nlp_f<F:
                improvement = True
                theta = theta_new
                lam = max(lam/10,1e-07)
            else:
                lam = min(lam*10,1e07)
                
        nlp_v = nlp_val(*list(theta.values()))
        
        if nlp_v < nlp_val_hist:
            nlp_val_hist = nlp_v
        else:
            print('Validation loss increased. Stopping optimization.')
            break
            
   
    return theta,nlp_train,nlp_v

def parallel_mode(model,u,y_ref,x0,switch=None,params=None):
      
    loss = 0
    y = []
    x = []
    e = []
    # Loop over all batches 
    for i in range(0,len(u)):   
        
        try:
            model.switching_instances = switch[i]
        except TypeError:
            pass

        # Simulate Model
        pred = model.Simulation(x0[i],u[i],params)
        

        if isinstance(pred, tuple):           
            x.append(pred[0])
            y.append(pred[1])
        else:
            y.append(pred)
        # Calculate simulation error            
        # Check for the case, where only last value is available
        
        if y_ref[i].shape[0]==1:
            y[-1]=y[-1][-1,:]
            e.append(y_ref[i] - y[-1])
            loss = loss + cs.sumsqr(e[-1])
    
        else:
            e.append(y_ref[i] - y[-1])
            loss = loss + cs.sumsqr(e[-1])
    
    return loss,e,x,y

def static_mode(model,u,y_ref,params=None):
    
    loss = 0
    y = []
    e = []
    
                    
    # Loop over all batches 
    for i in range(0,len(u)): 

        # One-Step prediction
        for k in range(u[i].shape[0]):  
            # print(k)
            y_new = model.OneStepPrediction(u[i][k,:],params)
            # print(y_new)
            y.append(y_new)
            e.append(y_ref[i][k,:]-y_new)
            # Calculate one step prediction error
            loss = loss + cs.sumsqr(e[-1]) 
            
        break
    
    return loss,e,y


def series_parallel_mode(model,u,y_ref,x_ref,x0,params=None):
   
    loss = 0
    y = []
    x = []

    
    # Training in series parallel configuration        
    # Loop over all batches 
    for i in range(0,len(u)):  
        x_batch = np.zeros(u[i,:,:].shape[0],model.dim_c)
        y_batch = np.zeros(u[i,:,:].shape[0],model.dim_y)
        
        # One-Step prediction
        for k in range(u[i,:,:].shape[0]-1):  
            # predict x1 and y1 from x0 and u0
            x_new,y_new = model.OneStepPrediction(x_ref[i,k,:],u[i,k,:],
                                                  params)
            x_batch[k,:] = x_new
            y_batch[k,:] = y_new
            
            # Calculate one step prediction error as 
            loss = loss + cs.sumsqr(y_ref[i,k,:]-y_new) + \
                cs.sumsqr(x_ref[i,k+1,:]-x_new) 
        
        x.append(x_batch)
        y.append(y_batch)
        
                
    return loss,x,y 


def AddParameterUpdate(parameter_dict,update):
    '''
    Adds an increment to model parameters

    Parameters
    ----------
    update : array like, vector
        DESCRIPTION.

    Returns
    -------
    None.       
    '''                    

    # Create empty dictionary
    Parameters_new = {}
            
    c = 0
    
    for param in parameter_dict.keys():
        dim0 = parameter_dict[param].shape[0]
        dim1 = parameter_dict[param].shape[1]
        
        Parameters_new[param] = parameter_dict[param] + \
            update[c:c+dim0*dim1].reshape((dim0,dim1))
                    
        c = c + dim0*dim1
        
        
    return Parameters_new