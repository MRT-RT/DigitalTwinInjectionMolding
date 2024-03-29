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


# from DIM.optim.DiscreteBoundedPSO import DiscreteBoundedPSO
from .common import OptimValues_to_dict,BestFitRate

import multiprocessing

class ParamOptimizer():
    """
    Offline Optimizer for models of type Recurrent, Static, QualityModel
    """

    def __init__(self,model,data_train,data_val,**kwargs):
        
        self.model = model
        self.data_train = data_train
        self.data_val = data_val
        
        self.initializations = kwargs.pop('initializations',10)   
        self.s_opts = kwargs.pop('s_opts',{"max_iter": 1000, "print_level":1,
                                           "hessian_approximation":'limited-memory'})
        self.p_opts = kwargs.pop('s_opts',{"expand":False})
        self.mode = kwargs.pop('mode','parallel') 
        self.n_pool = kwargs.pop('n_pool',None)     
        self.constraints = kwargs.pop('constraints',None)
        
        self.res_path = kwargs.pop('res_path',None)
        
        

    def optimize(self):
        
        results = [] 
        
        if self.n_pool is None:
            for i in range(0,self.initializations):
                
                if i > 0:
                    self.model.ParameterInitialization()
                
                res = self.param_est(self.model,self.data_train,self.data_val,
                                     self.p_opts,self.s_opts,self.mode,
                                     self.constraints)
                       
                # save parameters and performance in list
                results.append(res)
                   
            results = pd.DataFrame(data = results, columns = ['params_train',
                                'params_val','loss_train','loss_val','bfr_val'])
        else:

            data_train = [copy.deepcopy(self.data_train) for i in range(0,self.initializations)]
            data_val = [copy.deepcopy(self.data_val) for i in range(0,self.initializations)]
            models = [copy.deepcopy(self.model) for i in range(0,self.initializations)]
            p_opts = [copy.deepcopy(self.p_opts) for i in range(0,self.initializations)]
            s_opts = [copy.deepcopy(self.s_opts) for i in range(0,self.initializations)]
            mode = [copy.deepcopy(self.mode) for i in range(0,self.initializations)]
            constraints = [copy.deepcopy(self.constraints) for i in range(0,self.initializations)]
            
            # Hilft nicht:
            # models = [model.__class__(**model.__dict__) for i in range(0,initializations)]
            
            # Hilft nicht:    
            # Bug Fix: For some reason, although models are reinitialized in 
            # TrainingProcedure, exactly n_pool models ended up with the exact
            # same parameters. Therefore models are reinitialized here again before 
            # processes are created
            
            for model in models:
                model.ParameterInitialization()
            
            
            pool = multiprocessing.Pool(self.n_pool)
            results = pool.starmap(self.param_est, zip(models, data_train, data_val, 
                                                          p_opts, s_opts, mode,
                                                          constraints))        
            results = pd.DataFrame(data = results, columns = ['params_train',
                                'params_val','loss_train','loss_val','bfr_val'])
            
            pool.close() 
            pool.join()
            
            if self.res_path is not None:
                
                if not os.path.exists(self.res_path):
                    os.makedirs(self.res_path)
                    print('Created directory for results')
                    
                pkl.dump(results,open((self.res_path/'overview.pkl').as_posix(),
                                      'wb'))
                
                # Assign parameters to model and save in a dictionary
                models = {}
                for i in results.index:
                    models[i] = {}
                    models[i]['train'] = copy.deepcopy(model)
                    models[i]['train'].Parameters = results.loc[i,'params_train']
                    models[i]['val'] = copy.deepcopy(model)
                    models[i]['val'].Parameters = results.loc[i,'params_val']
                    
                pkl.dump(models,open((self.res_path/'models.pkl').as_posix(),
                                      'wb'))
                    
                
                
                
        return results

    def param_est(self,model,data_train,data_val,p_opts,s_opts,mode,
                         constraints):
              
        # # Create dictionary of all non-frozen parameters to create Opti Variables of 
        # OptiParameters = self.model.Parameters.copy()
        
        # for frozen_param in self.model.frozen_params:
        #     OptiParameters.pop(frozen_param)
            
        
        # params_opti = CreateOptimVariables(opti, OptiParameters)   
        
        # Evaluate on model on data
        
        # Create Instance of the Optimization Problem
        opti = cs.Opti()
        
        # Create dictionary of all non-frozen parameters to create Opti Variables of 
        OptiParameters = model.Parameters.copy()
        
        for frozen_param in model.frozen_params:
            OptiParameters.pop(frozen_param)
        
        
        opti_vars = self.__create_opti_vars(opti,OptiParameters)  
        
                
        if mode == 'parallel':
            f_eval =  model.parallel_mode
        elif mode == 'static':
            f_eval =  model.static_mode
        elif mode == 'series':
            f_eval =  model.series_parallel_mode
           
        loss_train,_ = f_eval(data_train,opti_vars)
        loss_val,_ = f_eval(data_val,opti_vars)
            
        
        
        loss_val = cs.Function('loss_val',[*list(opti_vars.values())],
                             [loss_val],list(opti_vars.keys()),['F'])
         
        opti.minimize(loss_train)
            
        # Create Solver
        opti.solver("ipopt",p_opts, s_opts)
        
        
        class intermediate_results():
            def __init__(self):
                self.F_val = np.inf
                self.params_val = {}
                
            def callback(self,i):
                params_val_new = OptimValues_to_dict(opti_vars,opti.debug)
                
                F_val_new = loss_val(*list(params_val_new.values()))

                if F_val_new < self.F_val:
                    self.F_val = F_val_new
                    self.params_val = params_val_new
                    print('Validation loss: ' + str(self.F_val))
        
        val_results = intermediate_results()

        # Callback
        opti.callback(val_results.callback)
        
        # Add constraints to optimization problem
        if constraints is not None:
            opti = self.__add_opti_constraints(opti,opti_vars,constraints) 

        # Set initial values of Opti Variables as current Model Parameters
        for key in opti_vars:
            opti.set_initial(opti_vars[key], model.Parameters[key])

        # Solve NLP, if solver does not converge, use last solution from opti.debug
        try: 
            sol = opti.solve()
        except:
            sol = opti.debug
            
        params = OptimValues_to_dict(opti_vars,sol)
        F_train = sol.value(opti.f)        

        params_val = val_results.params_val
        F_val = val_results.F_val
        
        
        # Assign parameters to model and evaluate
        model.Parameters = params
        
        _,y_est = f_eval(data_val)
        
        bfr = BestFitRate(data_val[model.y_label],y_est[model.y_label])
        print(bfr)
        
        return params,params_val,float(F_train),float(F_val),float(bfr)     
        
    def __create_opti_vars(self,opti,OptiParameters):
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
        
        for param in OptiParameters.keys():
            dim0 = OptiParameters[param].shape[0]
            dim1 = OptiParameters[param].shape[1]
            
            opti_vars[param] = opti.variable(dim0,dim1)
        
        
        return opti_vars
    
def __add_opti_constraints(self,opti,params_opti,constraints):
    
    for constraint in constraints:
        
        expression = "params_opti['"+constraint[0]+"']" + constraint[1]        
        opti.subject_to(eval(expression))

    
    return opti

# def ControlInput(ref_trajectories,opti_vars,k):
#     """
#     Übersetzt durch Maschinenparameter parametrierte
#     Führungsgrößenverläufe in optimierbare control inputs
#     """
    
#     control = []
            
#     for key in ref_trajectories.keys():
#         control.append(ref_trajectories[key](opti_vars,k))
    
#     control = cs.vcat(control)

#     return control   
    
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

# def CreateSymbolicVariables(Parameters):
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
#     opti_vars_vec = []
#     for param in Parameters.keys():
#         dim0 = Parameters[param].shape[0]
#         dim1 = Parameters[param].shape[1]
        
#         opti_vars[param] = cs.MX.sym(param,dim0,dim1)
#         # opti_vars_vec.append(opti_vars[param].reshape((-1,1)))
        
#     return opti_vars

# def AddOptimConstraints(opti,params_opti,constraints):
    
#     for constraint in constraints:
        
#         expression = "params_opti['"+constraint[0]+"']" + constraint[1]        
#         opti.subject_to(eval(expression))

    
#     return opti

# def ModelTraining(model,data_train,data_val,initializations=10, BFR=False, 
#                   p_opts=None, s_opts=None,mode='parallel',**kwargs):
    
   
#     results = [] 
    
#     for i in range(0,initializations):
        
#         # Initialize with the current model parameter during first initiali-
#         # zation and randomly during subsequent initializations
#         if i > 0:
#             model.ParameterInitialization()
#         res = TrainingProcedure(model, data_train,data_val, p_opts, s_opts,
#                                 mode,**kwargs)
               
#         # save parameters and performance in list
#         results.append(res)
           
#     results = pd.DataFrame(data = results, columns = ['loss_train','loss_val',
#                         'model','params_train','params_val'])
#     return results

# def TrainingProcedure(model, data_train, data_val, p_opts, s_opts, 
#                       mode,**kwargs):
    
#     # initialize model to make sure given initial parameters are assigned
    
#     # Auskommen tieren hilft! 
#     # model.ParameterInitialization()
    
#     # Estimate Parameters on training data
#     params_train,params_val,loss_train,loss_val = \
#         ModelParameterEstimation(model,data_train,data_val,p_opts,s_opts,mode,
#                                  **kwargs)
    
#     # save parameters and performance in list
#     result = [loss_train,loss_val,model.name,params_train,params_val]
    
#     return result

# def ParallelModelTraining(model,data_train,data_val,initializations=10,
#                           BFR=False, p_opts=None, s_opts=None,mode='parallel',
#                           n_pool=5):
    
     
#     data_train = [copy.deepcopy(data_train) for i in range(0,initializations)]
#     data_val = [copy.deepcopy(data_val) for i in range(0,initializations)]
#     models = [copy.deepcopy(model) for i in range(0,initializations)]
#     p_opts = [copy.deepcopy(p_opts) for i in range(0,initializations)]
#     s_opts = [copy.deepcopy(s_opts) for i in range(0,initializations)]
#     mode = [copy.deepcopy(mode) for i in range(0,initializations)]
    
#     # Hilft nicht:
#     # models = [model.__class__(**model.__dict__) for i in range(0,initializations)]
    
#     # Hilft nicht:    
#     # Bug Fix: For some reason, although models are reinitialized in 
#     # TrainingProcedure, exactly n_pool models ended up with the exact
#     # same parameters. Therefore models are reinitialized here again before 
#     # processes are created
    
#     for model in models:
#         model.ParameterInitialization()
    
    
#     pool = multiprocessing.Pool(n_pool)
#     results = pool.starmap(TrainingProcedure, zip(models, data_train, data_val, 
#                                                   p_opts, s_opts, mode))        
#     results = pd.DataFrame(data = results, columns = ['loss_train','loss_val',
#                         'model','params_train','params_val'])
    
#     pool.close() 
#     pool.join()      
    
#     return results 

# def HyperParameterPSO(model,data,param_bounds,n_particles,options,
#                       initializations=10,p_opts=None,s_opts=None):
#     """
#     Binary PSO for optimization of Hyper Parameters such as number of layers, 
#     number of neurons in hidden layer, dimension of state, etc

#     Parameters
#     ----------
#     model : model
#         A model whose hyperparameters to be optimized are attributes of this
#         object and whose model equations are implemented as a casadi function.
#     data : dict
#         A dictionary with training and validation data, see ModelTraining()
#         for more information
#     param_bounds : dict
#         A dictionary with structure {'name_of_attribute': [lower_bound,upper_bound]}
#     n_particles : int
#         Number of particles to use
#     options : dict
#         options for the PSO, see documentation of toolbox.
#     initializations : int, optional
#         Number of times the nonlinear optimization problem is solved for 
#         each particle. The default is 10.
#     p_opts : dict, optional
#         options to give to the optimizer, see Casadi documentation. The 
#         default is None.
#     s_opts : dict, optional
#         options to give to the optimizer, see Casadi documentation. The 
#         default is None.

#     Returns
#     -------
#     hist, Pandas Dataframe
#         Returns Pandas dataframe with the loss associated with each particle 
#         in the first column and the corresponding hyperparameters in the 
#         second column

#     """
    
#     path = 'temp/PSO_param/'
    
#     # Formulate Particle Swarm Optimization Problem
#     dimensions_discrete = len(param_bounds.keys())
#     lb = []
#     ub = []
    
#     for param in param_bounds.keys():
        
#         lb.append(param_bounds[param][0])
#         ub.append(param_bounds[param][1])
    
#     bounds= (lb,ub)
    
#     # Define PSO Problem
#     PSO_problem = DiscreteBoundedPSO(n_particles, dimensions_discrete, 
#                                      options, bounds)

#     # Make a directory and file for intermediate results 
#     try:
#         os.makedirs(path+model.name)
    
#         for key in param_bounds.keys():
#             param_bounds[key] = np.arange(param_bounds[key][0],
#                                           param_bounds[key][1]+1,
#                                           dtype = int)
        
#         index = pd.MultiIndex.from_product(param_bounds.values(),
#                                            names=param_bounds.keys())
        
#         hist = pd.DataFrame(index = index, columns=['cost','model_params'])    
        
#         pkl.dump(hist, open(path + model.name +'/' + 'HyperParamPSO_hist.pkl','wb'))
    
#     except:
#         print('Found data, PSO continues...')
        
#     # Define arguments to be passed to vost function
#     cost_func_kwargs = {'model': model,
#                         'param_bounds': param_bounds,
#                         'n_particles': n_particles,
#                         'dimensions_discrete': dimensions_discrete,
#                         'initializations':initializations,
#                         'p_opts': p_opts,
#                         's_opts': s_opts,
#                         'path':path}
    
#     # Create Cost function
#     def PSO_cost_function(swarm_position,**kwargs):
        
#         # Load training history to avoid calculating stuff muliple times
#         hist = pkl.load(open(path+ model.name +'/' +
#                              'HyperParamPSO_hist.pkl','rb'))
            
       
#         # Initialize empty array for costs
#         cost = np.zeros((n_particles,1))
    
#         for particle in range(0,n_particles):
            
#             # Check if results for particle already exist in hist
#             idx = tuple(swarm_position[particle].tolist())
            
#             if (math.isnan(hist.loc[idx,'cost']) and
#             math.isnan(hist.loc[idx,'model_params'])):
                
#                 # Adjust model parameters according to particle
#                 for p in range(0,dimensions_discrete):  
#                     setattr(model,list(param_bounds.keys())[p],
#                             swarm_position[particle,p])
                
#                 model.Initialize()
                
#                 # Estimate parameters
#                 results = ModelTraining(model,data,initializations, 
#                                         BFR=False, p_opts=p_opts, 
#                                         s_opts=s_opts)
                
#                 # Save all results of this particle in a file somewhere so that
#                 # the nonlinear optimization does not have to be done again
                
#                 pkl.dump(results, open(path + model.name +'/' + 'particle' + 
#                                        str(swarm_position[particle]) + '.pkl',
#                                        'wb'))
                
#                 # calculate best performance over all initializations
#                 cost[particle] = results.loss_val.min()
                
#                 # Save new data to dictionary for future iterations
#                 hist.loc[idx,'cost'] = cost[particle]
                
#                 # Save model parameters corresponding to best performance
#                 idx_min = results['loss_val'].idxmin()
#                 hist.loc[idx,'model_params'] = \
#                 [results.loc[idx_min,'params']]
                
#                 # Save DataFrame to File
#                 pkl.dump(hist, open(path + model.name +'/' +
#                                     'HyperParamPSO_hist.pkl','wb'))
                
#             else:
#                 cost[particle] = hist.loc[idx].cost.item()
                
        
        
        
#         cost = cost.reshape((n_particles,))
#         return cost
    
    
#     # Solve PSO Optimization Problem
#     PSO_problem.optimize(PSO_cost_function, iters=100, n_processes=None,**cost_func_kwargs)
    
#     # Load intermediate results
#     hist = pkl.load(open(path + model.name +'/' + 'HyperParamPSO_hist.pkl','rb'))
    
#     # Delete file with intermediate results
#     # os.remove(path + model.name +'/' + 'HyperParamPSO_hist.pkl')
    
#     return hist

# def ModelParameterEstimationLM(model,data,p_opts=None,s_opts=None,mode='parallel'):
#     """
    

#     Parameters
#     ----------
#     model : model
#         A model whose hyperparameters to be optimized are attributes of this
#         object and whose model equations are implemented as a casadi function.
#     data : dict
#         A dictionary with training and validation data, see ModelTraining()
#         for more information
#     p_opts : dict, optional
#         options to give to the optimizer, see Casadi documentation. The 
#         default is None.
#     s_opts : dict, optional
#         options to give to the optimizer, see Casadi documentation. The 
#         default is None.

#     Returns
#     -------
#     values : dict
#         dictionary with either the optimal parameters or if the solver did not
#         converge the last parameter estimate

#     """
    
#     max_iter = s_opts['max_iter']
#     step = s_opts['step']
    
#     # Create dictionary of all non-frozen parameters to create Opti Variables of 
#     # OptiParameters = model.Parameters.copy()
#     params_opti = CreateSymbolicVariables(model.Parameters)      
    
#     # Evaluate on model on data
#     u_train = data['u_train']
#     y_ref_train = data['y_train']

#     u_val = data['u_val']
#     y_ref_val = data['y_val']
    
#     try:
#         switch_train =  data['switch_train']
#         switch_val =  data['switch_val']
#     except KeyError:
#         switch = None
    
#     if mode == 'parallel':
#         x0_train = data['init_state_train']
#         x0_val = data['init_state_val']
        
#         loss_train,_,_,_ = model.parallel_mode(u_train,y_ref_train,x0_train,
#                                          switch_train,params_opti) 
        
#         loss_val,_,_,_ = model.parallel_mode(u_val,y_ref_val,x0_val,
#                                        switch_val,params_opti)
        
#     elif mode == 'static':
#         loss_train,_,_ = model.static_mode(u_train,y_ref_train,params_opti)   
#         loss_val,_,_ = model.static_mode(u_val,y_ref_val,params_opti) 
                
#     elif mode == 'series':
#         x0 = data['init_state_train']
#         loss_train,_,_ = model.series_parallel_mode(u,y_ref,x0,params_opti)
    
     
    
#     opti_vars_vec = cs.vcat([params_opti[p].reshape((-1,1)) for p in 
#                              params_opti.keys() if p not in model.frozen_params])
    
#     # LM Optimizer
#     grad = cs.gradient(loss_train,opti_vars_vec)
#     hess = cs.mtimes(grad,grad.T)

#     train = cs.Function('loss_train',[*list(params_opti.values())],
#                          [loss_train,grad,hess],list(params_opti.keys()),
#                          ['F','G','H'])
#     val = cs.Function('loss_val',[*list(params_opti.values())],
#                          [loss_val],list(params_opti.keys()),['F'])
    
    
#     # replace frozen parameters in params_opti with numeric model parameters
#     # for p in model.frozen_params:
#     #     params_opti[p] = model.Parameters[p]      

#     lam = 1
#     params = model.Parameters.copy()
    
#     nlp_val_hist = np.inf
    
#     for i in range(0,max_iter):

#         FGH = train(**params)
#         F_val = val(**params)
        
#         F = FGH['F']
#         G = FGH['G']
#         H = FGH['H']  
        
#         F_val = F_val['F']
        
#         print('Iteration: '+str(i) + '   loss_train: ' + str(F) + \
#               '   loss_val: ' + str(F_val) + '   lambda:' + str(lam))
            

        
#         improvement = False
        
#         while improvement is False:
            
#             d_theta = -step*cs.mtimes(cs.inv(H+lam*np.eye(H.shape[0])),G)*F
            
#             # new params
#             params_new =  AddParameterUpdate(params,d_theta,
#                                             model.frozen_params)
            
#             # evaluate loss
#             f = train(**params_new)['F']
#             v = val(**params_new)['F']
            
#             if f<F:
#                 improvement = True
#                 params = params_new
#                 lam = max(lam/10,1e-10)
#             elif lam == 1e10:
#                 print('Keine Verbesserung möglich, breche Optimierung ab!')
#                 break
#             else:                    
#                 lam = min(lam*10,1e10)
                 
        
#         if v < nlp_val_hist:
#             nlp_val_hist = v
#             params_save = params.copy()

            
            
#     return params,params_save,float(F),float(F_val)



# def ModelParameterEstimation(model,data_train,data_val,p_opts=None,
#                              s_opts=None,mode='parallel',**kwargs):
#     """
    

#     Parameters
#     ----------
#     model : model
#         A model whose hyperparameters to be optimized are attributes of this
#         object and whose model equations are implemented as a casadi function.
#     data : dict
#         A dictionary with training and validation data, see ModelTraining()
#         for more information
#     p_opts : dict, optional
#         options to give to the optimizer, see Casadi documentation. The 
#         default is None.
#     s_opts : dict, optional
#         options to give to the optimizer, see Casadi documentation. The 
#         default is None.

#     Returns
#     -------
#     values : dict
#         dictionary with either the optimal parameters or if the solver did not
#         converge the last parameter estimate

#     """
#     # Create Instance of the Optimization Problem
#     opti = cs.Opti()
    
#     # Create dictionary of all non-frozen parameters to create Opti Variables of 
#     OptiParameters = model.Parameters.copy()
    
#     for frozen_param in model.frozen_params:
#         OptiParameters.pop(frozen_param)
        
    
#     params_opti = CreateOptimVariables(opti, OptiParameters)   
    
#     # Evaluate on model on data
    
#     if mode == 'parallel':
#         loss_train,_ = model.parallel_mode(data_train,params_opti)
#         loss_val,_ = model.parallel_mode(data_val,params_opti)
        
#     elif mode == 'static':
#         loss_train,_ = model.static_mode(data_train,params_opti)  
#         loss_val,_ = model.static_mode(data_val,params_opti)
                
#     elif mode == 'series':      
#         loss_train,_ = model.series_parallel_mode(data_train,params_opti)
#         loss_val,_ = model.series_parallel_mode(data_val,params_opti)
    
#     loss_val = cs.Function('loss_val',[*list(params_opti.values())],
#                          [loss_val],list(params_opti.keys()),['F'])
     
#     opti.minimize(loss_train)

#     # Solver options
#     if p_opts is None:
#         p_opts = {"expand":False}
#     if s_opts is None:
#         s_opts = {"max_iter": 1000, "print_level":1}


        
#     # Create Solver
#     opti.solver("ipopt",p_opts, s_opts)
        
#     class intermediate_results():
#         def __init__(self):
#             self.F_val = np.inf
#             self.params_val = {}
            
#         def callback(self,i):
#             params_val_new = OptimValues_to_dict(params_opti,opti.debug)
            
#             F_val_new = loss_val(*list(params_val_new.values()))

#             if F_val_new < self.F_val:
#                 self.F_val = F_val_new
#                 self.params_val = params_val_new
#                 print('Validation loss: ' + str(self.F_val))
    
#     val_results = intermediate_results()

#     # Callback
#     opti.callback(val_results.callback)
    
    
#     # Constraints
#     constraints = kwargs.pop('constraints',None)
    
#     if constraints is not None:
#         opti = AddOptimConstraints(opti,params_opti,constraints) 

    
    
#     # Set initial values of Opti Variables as current Model Parameters
#     for key in params_opti:
#         opti.set_initial(params_opti[key], model.Parameters[key])

#     # Solve NLP, if solver does not converge, use last solution from opti.debug
#     try: 
#         sol = opti.solve()
#     except:
#         sol = opti.debug
        
#     params = OptimValues_to_dict(params_opti,sol)
#     F_train = sol.value(opti.f)        

#     params_val = val_results.params_val
#     F_val = val_results.F_val
    
            
#     return params,params_val,float(F_train),float(F_val)


    
# def parallel_mode(model,data,params=None):
      
#     loss = 0

    
#     simulation = []
    
#     # Loop over all batches 
#     for i in range(0,len(data['data'])):
        
#         io_data = data['data'][i]
#         x0 = data['init_state'][i]
#         try:
#             switch = data['switch'][i]
#             # switch = [io_data.index.get_loc(s) for s in switch]
#             kwargs = {'switching_instances':switch}
                        
#             # print('Kontrolliere ob diese Zeile gewünschte Indizes zurückgibt!!!')               
#         except KeyError:
#             switch = None
        
        
#         u = io_data.iloc[0:-1][model.u_label]
#         # u = io_data[model.u_label]

        
#         # Simulate Model        
#         pred = model.Simulation(x0,u,params,**kwargs)
        
        
#         y_ref = io_data[model.y_label].values
        
        
#         if isinstance(pred, tuple):           
#             x_est= pred[0]
#             y_est= pred[1]
#         else:
#             y_est = pred
            
#         # Calculate simulation error            
#         # Check for the case, where only last value is available
        
#         if np.all(np.isnan(y_ref[1:])):
            
#             y_ref = y_ref[[0]]
#             y_est=y_est[-1,:]
#             e= y_ref - y_est
#             loss = loss + cs.sumsqr(e)
            
#             idx = [i]
    
#         else :
            
#             y_ref = y_ref[1:1+y_est.shape[0],:]                                                 # first observation cannot be predicted
            
#             e = y_ref - y_est
#             loss = loss + cs.sumsqr(e)
            
#             idx = io_data.index
        
#         if params is None:
#             y_est = np.array(y_est)
            
#             df = pd.DataFrame(data=y_est, columns=model.y_label,
#                               index=idx)
            
#             simulation.append(df)
#         else:
#             simulation = None
            
#     return loss,simulation

# def static_mode(model,data,params=None):
    
    
#     y_est = []
   
#     # u = data[model.u_label].values
#     u = data[model.u_label]
    
#     # If parameters are not given only calculate model prediction    
#     if params is None:
        
#         # One-Step prediction
#         for k in range(u.shape[0]):  
#             # y_new = model.OneStepPrediction(u[k,:],params)
#             y_new = model.OneStepPrediction(u.iloc[[k]],params)
            
#             y_est.append(y_new)
        
#         y_est = np.array(y_est).reshape((-1,len(model.y_label)))
        
        
#         # Rename columns to distinguish estimate from true value
#         # cols = [label + '_est' for label in model.y_label]
        
#         cols = model.y_label
        
#         df = pd.DataFrame(data=y_est, columns=cols, index=data.index)
        
#         loss = None
    
#     # else calulate loss
#     else:
        
#         y_ref = data[model.y_label].values
        
#         loss = 0
#         e = [] 
#         # One-Step prediction
#         for k in range(u.shape[0]):
#             y_new = model.OneStepPrediction(u.iloc[[k]],params)
#             # y_new = model.OneStepPrediction(u[k,:],params)
#             y_est.append(y_new)
#             e.append(y_ref[k,:]-y_new)
#             loss = loss + cs.sumsqr(e[-1])
            
#         df = None
        
#     return loss,df


# def series_parallel_mode(model,data,params=None):
  
#     loss = 0
    
#     x = []
    
#     prediction = []

#     # if None is not None:
        
#     #     print('This is not implemented properly!!!')
        
#     #     for i in range(0,len(u)):
#     #         x_batch = []
#     #         y_batch = []
            
#     #         # One-Step prediction
#     #         for k in range(u[i].shape[0]-1):
                
                
                
#     #             x_new,y_new = model.OneStepPrediction(x_ref[i][k,:],u[i,k,:],
#     #                                                   params)
#     #             x_batch.append(x_new)
#     #             y_batch.append(y_new)
                
#     #             loss = loss + cs.sumsqr(y_ref[i][k,:]-y_new) + \
#     #                 cs.sumsqr(x_ref[i,k+1,:]-x_new) 
            
#     #         x.append(x_batch)
#     #         y.append(y_batch)
        
#     #     return loss,x,y 
    
#     # else:
#     for i in range(0,len(data['data'])):
        
#         io_data = data['data'][i]
#         x0 = data['init_state'][i]
#         switch = data['switch'][i]
        
#         y_est = []
#         # One-Step prediction
#         for k in range(0,io_data.shape[0]-1):
            
#             uk = io_data.iloc[k][model.u_label].values.reshape((-1,1))
#             yk = io_data.iloc[k][model.y_label].values.reshape((-1,1))
            
#             ykplus = io_data.iloc[k+1][model.y_label].values.reshape((-1,1))
            
#             # predict x1 and y1 from x0 and u0
#             y_new = model.OneStepPrediction(yk,uk,params)
            
#             loss = loss + cs.sumsqr(ykplus-y_new)        
            
#             y_est.append(y_new.T)
        
#         y_est = cs.vcat(y_est)
        
#         if params is None:
#             y_est = np.array(y_est)
            
#             df = pd.DataFrame(data=y_est, columns=model.y_label,
#                               index=io_data.index[1::])
            
#             prediction.append(df)
#         else:
#             prediction = None
        
#     return loss,prediction

# def AddParameterUpdate(parameter_dict,update,frozen_parameters=[]):
#     '''
#     Adds an increment to model parameters

#     Parameters
#     ----------
#     update : array like, vector
#         DESCRIPTION.

#     Returns
#     -------
#     None.       
#     '''                    

#     # Create empty dictionary
#     Parameters_new = parameter_dict.copy()
            
#     c = 0
    
#     for param in parameter_dict.keys():
        
#         if param not in frozen_parameters:
#             dim0 = parameter_dict[param].shape[0]
#             dim1 = parameter_dict[param].shape[1]
        
#             Parameters_new[param] = parameter_dict[param] + \
#                 update[c:c+dim0*dim1].reshape((dim0,dim1))
                    
#             c = c + dim0*dim1
        
        
#     return Parameters_new

