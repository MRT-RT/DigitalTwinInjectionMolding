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

class Optimizer():
    '''
    Parent class for all optimizers with basic functionality
    ''' 
    
    def AddOptimConstraints(self,opti,params_opti,constraints):
        """
        Adds constraints that are given as a list of tuples to an casadi
        opti instance
        
        Parameters
        ----------
        opti : instance of casadi opti class
        constraints : list of tuples
            Each tuple has the form
            ('<name_of_opti_variable>','<operator><value>')
            e.g. ('Einspritzgeschwindigkeit','<10.0')

        Returns
        -------
        opti : instance of opti class
            instance of opti class updated with constraints
        """
        for constraint in constraints:
            
            expression = "params_opti['"+constraint[0]+"']" + constraint[1]        
            opti.subject_to(eval(expression))
    
        
        return opti



class StaticProcessOptimizer(Optimizer):
    
    def __init__(self,model):
        """
        Defines the loss function to be minimized during numerical optimal
        control as casadi function

        Parameters
        ----------
        model : DIM.models.model_structures.Static
            Instance of Static

        Returns
        -------
        None.

        """
        
        self.model = model
        
        # Define the function to be minimized as casadi function
        # define inputs as symbolic variables
        U = [cs.MX.sym(u,1,1) for u in model.u_label]
        Q = [cs.MX.sym(y,1,1) for y in model.y_label]
        
        
        Q_pred = model.OneStepPrediction(cs.vcat(U))
        
        loss = cs.sumsqr(Q_pred-cs.vcat(Q))
        
        self.LossFunc = cs.Function('loss',U+Q, [loss], 
                                    model.u_label+model.y_label,['loss'])
        
        
        
    def optimize(self,Q_target,fix_inputs,**kwargs):
        """
        

        Parameters
        ----------
        Q_target : pd.DataFrame
            Reference output for numerical control problem.
        fix_inputs : pd.DataFrame
            If any inputs should not be optimized, provide numerical values
            for them here.
        **kwargs : dict
            input_init: pd.DataFrame with initial values for optimization problem
            constraints: list of tuples with constraints, see AddOptimConstraints
                            for documentation

        Returns
        -------
        U_sol : pd.DataFrame
            Optimal values found by IPOPT

        """
        
        
        input_init = kwargs.pop('input_init',None)
        constraints = kwargs.pop('constraints',None)
        
        # Recast DataFrames as Dictionaries
        Q_target_dic = Q_target.to_dict(orient='records')[0]
        fix_inputs_dic = fix_inputs.to_dict(orient='records')[0]
        
        # Create Instance of the optimization problem
        opti = cs.Opti()       
        
        # create opti variables 
        U = {label:opti.variable() for label in self.model.u_label if \
             label not in fix_inputs_dic}

        # Add constraints
        if constraints is not None:
            opti = self.AddOptimConstraints(opti,U,constraints) 

        # Set initial values for decision variables
        if input_init is not None:
            
            # Cast into dictionary
            input_init = input_init.to_dict(orient='records')[0]
            
            for key,value in input_init.items():
                opti.set_initial(U[key],value)
            
        # inputs = [opti.variable for label in self.model.u_label]
            
        # Merge dictionaries for function call
        inputs = {}
        inputs.update(U)
        inputs.update(fix_inputs_dic)
        inputs.update(Q_target_dic)
        
        # Define self.LossFunc as target for minimization
        opti.minimize(self.LossFunc(**inputs)['loss'])
        
        #Choose solver
        opti.solver('ipopt')
                
        # Solve optmization problem
        sol = opti.solve()
        
        # Extract real values from solution       
        U_sol = {}
        
        # Extract loss from solution
        loss = sol.value(opti.f) 
        
        for key,value in U.items():
            U_sol[key] = [sol.value(value)]

        # Append fixed inputs    
        U_sol.update(fix_inputs_dic)
        
        U_sol = pd.DataFrame.from_dict(U_sol)
        
        
        U_sol['loss'] = loss
        
        
        return U_sol
        


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

def ProcessMultiStageOptimization(process_model,target):
    
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
    elif output_dims.count(output_dims[0]) != len(subsystems):
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
    for k in range(N):
        
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
    for relaxation''' 
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

def QualityMultiStageOptimization(quality_model,target):
    """
    Single-shooting procedure for optimization of process variables given a 
    desired target quality

    Parameters
    ----------
    quality_model : QualityModel
        Container for models mapping process variable trajectories to quality
        measurements.
    target : array-like
        A vector containing the desired values of the quality variables.

    Returns
    -------
    values : TYPE
        DESCRIPTION.

    """
    subsystems = quality_model.subsystems
    switching_instances = quality_model.switching_instances
    
    
    output_dims = [sys.dim_out for sys in subsystems]
    input_dims = [sys.dim_u for sys in subsystems]
    c_dims = [sys.dim_c for sys in subsystems]
    
    # Plausibility and Dimension Checks
    if len(switching_instances) != len(subsystems):
        print('Number of switching instances does not fit number of Subsystems!')
    elif output_dims.count(output_dims[0]) != len(subsystems):
        print('All Subsystems need to have the same output dimension')        
    elif input_dims.count(input_dims[0]) != len(subsystems):
        print('All Subsystems need to have the same input dimension')         
    elif c_dims.count(c_dims[0]) != len(subsystems):
        print('All Subsystems need to have the same dimension of the hidden state')
    elif output_dims[0] != target.shape[0]:
        print('Dimension of model output and target must match')
        
    # Create Instance of the Optimization Problem
    opti = cs.Opti()
    
    # Translate Maschinenparameter into opti.variables
    # ref_params_opti = CreateOptimVariables(opti, ref_params)
    
    # Number of time steps
    N = np.sum(switching_instances)
    
    # Create decision variables for states
    U = opti.variable(N,input_dims[0])
    
    # Create empty arrays for output Y and hidden state X
    Y = []
    X = []    
    
    # Initial hidden state
    X.append(np.zeros((c_dims[0],1)))
    
    # Initial Constraints
    # opti.subject_to(X[0]==target[0])
    
    # Generate an index vector pointing to the active subsystem in each time step
    active_subsystem = np.zeros(N,np.int8)

    for switch in np.cumsum(switching_instances)[:-1]:
        active_subsystem[switch::] = active_subsystem[switch::]+1
    
      
    # System Dynamics as Path Constraints
    for k in range(N-1):
        
        # Control input at every time step is a function of the parameters of
        # the reference trajectories 
        # U = ControlInput(reference[active_subsystem[k]],ref_params_opti,k)
        
        # Do a one step prediction based on the model
        pred = subsystems[active_subsystem[k]].OneStepPrediction(X[k],U[k,:])
        
        # OneStepPrediction can return a tuple, i.e. state and output. The 
        # output is per convention the second entry
        if isinstance(pred,tuple):
            X.append(pred[0])
            Y.append(pred[1])
        
        
            
    ''' Further Path Constraints (to avoid values that might damage the 
    machine or are in other ways harmful or unrealistic) '''
    
    # TO DO #
    
    
    ''' Final constraint might make solution infeasible, search for methods
    for relaxation''' 
    # opti.subject_to(X[-1]==target[-1])
    
    
    # Set initial values for process variables U THIS IS MAJOR WORK
    # for key in ref_params_opti:
        # opti.set_initial(ref_params_opti[key],ref_params[key])

   
    # Define Loss Function    
    opti.minimize(cs.sumsqr(Y[-1]-target))
    
    #Choose solver
    opti.solver('ipopt')
    
    # Get solution
    sol = opti.solve()
    
    # Extract real values from solution
    values = {}
    values['U'] = sol.value(U)
    # values['X'] = sol.value(X)
    # values['Y'] = sol.value(Y)
    
    return values

def SingleStageOptimization(QualityModel,ref):
    """ 
    single shooting procedure for optimal control of a scalar final value
    
    QualityModel: Quality Model
    ref: skalarer Referenzwert für Optimierungsproblem
    N: Anzahl an Zeitschritten
    """
    
    N = QualityModel.N
    model = QualityModel.model
    
    # Create Instance of the Optimization Problem
    opti = cs.Opti()
    
    # Create decision variables for states
    U = opti.variable(N,1)
        
    # Initial quality 
    x = np.zeros((model.dim_c,1))
    y = np.zeros((model.dim_out,1))
    X = [x]
    Y = [y]
    
    # Simulate Model
    for k in range(N):
        out = model.OneStepPrediction(X[k],U[k]) #SimulateModel(model,X[k],U[k],model.ModelParamsQuality)
        X.append(out[0])
        Y.append(out[1])
            
    X = cs.hcat(X)
    Y = cs.hcat(Y)
    
    # Define Loss Function  
    opti.minimize(cs.sumsqr(Y[-1]-ref))
                  
    #Choose solver
    opti.solver('ipopt')
    
    # Get solution
    sol = opti.solve()   

    # Extract real values from solution
    values = {}
    values['U'] = sol.value(U)
    
    return values






