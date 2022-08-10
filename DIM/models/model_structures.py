# -*- coding: utf-8 -*-

# from sys import path
# path.append(r"C:\Users\LocalAdmin\Documents\casadi-windows-py38-v3.5.5-64bit")
# import os
# print (os.getcwd())
import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from DIM.optim.common import RK4
from scipy.stats import ortho_group
from DIM.models.initializations import XavierInitialization, RandomInitialization, HeInitialization

# from miscellaneous import *
class static():
    """
    Base implementation of a static model.
    """
    def ParameterInitialization(self):
        '''
        Routine for parameter initialization. Takes input_names from the Casadi-
        Function defining the model equations self.Function and defines a 
        dictionary with input_names as keys. According to the initialization
        procedure defined in self.init_proc each key contains 
        a numpy array of appropriate shape

        Returns
        -------
        None.

        '''
                
        # Initialization procedure
        if self.init_proc == 'random':
            initialization = RandomInitialization
        elif self.init_proc == 'xavier':
            initialization = XavierInitialization
        elif self.init_proc == 'he':
            initialization = HeInitialization      
        
        # Define all parameters in a dictionary and initialize them 
        self.Parameters = {}
        
        new_param_values = {}
        for p_name in self.Function.name_in()[1::]:
            new_param_values[p_name] = initialization(self.Function.size_in(p_name))
        
        self.AssignParameters(new_param_values)

        # Initialize with specific inital parameters if given
        if self.initial_params is not None:
            for param in self.initial_params.keys():
                if param in self.Parameters.keys():
                    self.Parameters[param] = self.initial_params[param]

    def OneStepPrediction(self,u0,params=None):
        """
        OneStepPrediction() evaluates the model equation defined in 
        self.Function()
        
        self.Function() takes initial state x0, input u0 and all model 
        parameters as input. The model parameters can either be optimization
        variables themselves (as in system identification) or the take specific 
        values (when the estimated model is used for control)

        Parameters
        ----------
        u0 : array-like with dimension [self.dim_u, 1]
            input
        params : dictionary, optional
            params is None: This is the case during model based control,
            self.Function() is evaluated with the numerical
            values of the model parameters saved in self.Parameters
            params is dictionary of opti.variables: During system identification
            the model parameters are optimization variables themselves, so a 
            dictionary of opti.variables is passed to self.Function()

        Returns
        -------
        y : array-like with dimension [self.dim_x, 1]
            output of the Feedforward Neural Network

        """
        if params==None:
            params = self.Parameters
        
        params_new = []
            
        for name in  self.Function.name_in():
            try:
                params_new.append(params[name])                      # Parameters are already in the right order as expected by Casadi Function
            except:
                continue
        
        y = self.Function(u0,*params_new)     
                              
        return y
                      
    def AssignParameters(self,params):
        
        for p_name in self.Function.name_in()[1::]:
            self.Parameters[p_name] = params[p_name]


class recurrent():
    '''
    Parent class for all Models
    '''
    
    def ParameterInitialization(self):
        '''
        Routine for parameter initialization. Takes input_names from the Casadi-
        Function defining the model equations self.Function and defines a 
        dictionary with input_names as keys. According to the initialization
        procedure defined in self.init_proc each key contains 
        a numpy array of appropriate shape

        Returns
        -------
        None.

        '''
                
        # Initialization procedure
        if self.init_proc == 'random':
            initialization = RandomInitialization
        elif self.init_proc == 'xavier':
            initialization = XavierInitialization
        elif self.init_proc == 'he':
            initialization = HeInitialization      
        
        # Define all parameters in a dictionary and initialize them 
        self.Parameters = {}
        
        # new_param_values = {}
        for p_name in self.Function.name_in()[2::]:
            self.Parameters[p_name] = initialization(self.Function.size_in(p_name))
        
        # self.SetParameters(new_param_values)

        # Initialize with specific inital parameters if given
        # self.SetParameters(self.initial_params)
        if self.initial_params is not None:
            for param in self.initial_params.keys():
                if param in self.Parameters.keys():
                    self.Parameters[param] = self.initial_params[param]
        
        return None
                    
            
    def OneStepPrediction(self,x0,u0,params=None):
        '''
        Estimates the next state and output from current state and input
        x0: Casadi MX, current state
        u0: Casadi MX, current input
        params: A dictionary of opti variables, if the parameters of the model
                should be optimized, if None, then the current parameters of
                the model are used
        '''
        if params==None:
            params = self.Parameters
        
        params_new = []
            
        for name in self.Function.name_in()[2::]:
 
            try:
                params_new.append(params[name])                      # Parameters are already in the right order as expected by Casadi Function
            except:
                continue
                # params_new.append(self.Parameters[name])  
            
        x1,y1 = self.Function(x0,u0,*params_new)     
                              
                              
        return x1,y1       
           
   
    def Simulation(self,x0,u,params=None,**kwargs):
        '''
        A iterative application of the OneStepPrediction in order to perform a
        simulation for a whole input trajectory
        x0: Casadi MX, inital state a begin of simulation
        u: Casadi MX,  input trajectory
        params: A dictionary of opti variables, if the parameters of the model
                should be optimized, if None, then the current parameters of
                the model are used
        '''
        if params==None:
            params = self.Parameters
        
        params_new = []
            
        for name in self.Function.name_in()[2::]:
 
            try:
                params_new.append(params[name])                      # Parameters are already in the right order as expected by Casadi Function
            except:
                params_new.append(self.Parameters[name])
        
        u = u[self.u_label].values
        
        F_sim = self.Function.mapaccum(u.shape[0])
        # print(params_new)
        x,y = F_sim(x0,u.T,*params_new)
        
        x = x.T
        y = y.T

        return x,y    
    
    def SetParameters(self,params):
            
        for p_name in self.Function.name_in()[2::]:
            try:
                self.Parameters[p_name] = params[p_name]
            except:
                pass      

    # def Setinitial_params(self,initial_params):
    #     for p_name in self.Function.name_in()[2::]:
    #         try:
    #             self.initial_params[p_name] = initial_params[p_name]
    #         except:
    #             pass
            
    # def Setfrozen_params(self,frozen_params):
    #     # print(frozen_params)
    #     # print(self.Function.name_in()[2::])
    #     print(self.frozen_params)
    #     for p_name in self.Function.name_in()[2::]:
    #         if p_name in frozen_params:
    #             # print(p_name)
    #             self.frozen_params.append(p_name)
    #     print(self.frozen_params)

    
class State_MLP(recurrent):
    """
    Implementation of a single-layered Feedforward Neural Network.
    """

    def __init__(self,dim_u,dim_c,dim_hidden,dim_out,u_label,y_label,name,initial_params=None, 
                 frozen_params = [], init_proc='random'):
        """
        Initialization procedure of the Feedforward Neural Network Architecture
        
        Parameters
        ----------
        dim_u : int
            Dimension of the input, e.g. dim_u = 2 if input is a 2x1 vector
        dim_out : int
            Dimension of the output, e.g. dim_out = 3 if output is a 3x1 vector.
        dim_hidden : int
            Number of nonlinear neurons in the hidden layer, e.g. dim_hidden=10,
            if NN is supposed to have 10 neurons in hidden layer.
        u_label : 
        name : str
            Name of the model, e.g. name = 'InjectionPhaseModel'.

        Returns
        -------
        None.

        """
        self.dim_u = dim_u
        self.dim_c = dim_c
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        
        self.u_label = u_label
        self.y_label = y_label
        self.name = name
        
        self.initial_params = initial_params
        self.frozen_params = frozen_params
        self.init_proc = init_proc
        
        
        self.Initialize()

    def Initialize(self):
        """
        Defines the parameters of the model as symbolic casadi variables and 
        the model equation as casadi function. Model parameters are initialized
        randomly.

        Returns
        -------
        None.

        """   
        
        dim_u = self.dim_u
        dim_c = self.dim_c
        dim_hidden = self.dim_hidden
        dim_out = self.dim_out
        name = self.name      
    
        u = cs.MX.sym('u',dim_u,1)
        c = cs.MX.sym('c',dim_c,1)
        
        # Parameters
        # State equation parameters
        W_h = cs.MX.sym('W_h_'+name,dim_hidden,dim_u+dim_c)
        b_h = cs.MX.sym('b_h_'+name,dim_hidden,1)
        
        W_c = cs.MX.sym('W_c_'+name,dim_c,dim_hidden)
        b_c = cs.MX.sym('b_c_'+name,dim_c,1)

        # State equation parameters
        C = cs.MX.sym('C_'+name,dim_out,dim_c)


        # Model Equations
        h =  cs.tanh(cs.mtimes(W_h,cs.vertcat(u,c))+b_h)
        c_new = cs.mtimes(W_c,h)+b_c
        
        x_new = cs.mtimes(C,c_new)  
        
        input = [c,u,W_h,b_h,W_c,b_c,C]
        input_names = [var.name() for var in input]
        
        output = [c_new,x_new]
        output_names = ['c_new','x_new']
        
        self.Function = cs.Function(name, input, output, input_names,output_names)
        
        self.ParameterInitialization()
        
        return None

    

class TimeDelay_MLP(recurrent):
    """
    Implementation of a single-layered Feedforward Neural Network.
    """

    def __init__(self,dim_u,dim_hidden,dim_out,dim_c,u_label,y_label,name,initial_params=None, 
                 frozen_params = [], init_proc='xavier'):
        """
        Initialization procedure of the Feedforward Neural Network Architecture
        
        Parameters
        ----------
        dim_u : int
            Dimension of the input, e.g. dim_u = 2 if input is a 2x1 vector
        dim_out : int
            Dimension of the output, e.g. dim_out = 3 if output is a 3x1 vector.
        dim_hidden : int
            Number of nonlinear neurons in the hidden layer, e.g. dim_hidden=10,
            if NN is supposed to have 10 neurons in hidden layer.
        u_label : 
        name : str
            Name of the model, e.g. name = 'InjectionPhaseModel'.

        Returns
        -------
        None.

        """
        self.dim_u = dim_u
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        
        self.dim_c = dim_c
        
        self.u_label = u_label
        self.y_label = y_label
        self.name = name
        
        self.initial_params = initial_params
        self.frozen_params = frozen_params
        self.init_proc = init_proc
        
        self.dynamics = 'external'
        
        self.Initialize()

    def Initialize(self):
        """
        Defines the parameters of the model as symbolic casadi variables and 
        the model equation as casadi function. Model parameters are initialized
        randomly.

        Returns
        -------
        None.

        """   
        
        dim_u = self.dim_u
        dim_hidden = self.dim_hidden
        dim_out = self.dim_out
        dim_c = self.dim_c
        name = self.name      
    
        u = cs.MX.sym('u',dim_u*dim_c,1)
        c = cs.MX.sym('c',dim_out*dim_c,1)        
        
        # Parameters
        # State equation parameters
        W_h = cs.MX.sym('W_h_'+name,dim_hidden,(dim_u+dim_out)*dim_c)
        b_h = cs.MX.sym('b_h_'+name,dim_hidden,1)
        
        W_o = cs.MX.sym('W_o_'+name,dim_out,dim_hidden)
        b_o = cs.MX.sym('b_o_'+name,dim_out,1)




        # Model Equations
        h =  cs.tanh( cs.mtimes(W_h,cs.vertcat(u,c)) + b_h )    
        y_new = cs.mtimes(W_o,h)+b_o
        
        c_new = cs.vertcat(c,y_new)[dim_out::,:] 
        
        input = [c,u,W_h,b_h,W_o,b_o]
        input_names = [var.name() for var in input]
        
        output = [c_new,y_new]
        output_names = ['c_new','y_new']
        
        self.Function = cs.Function(name, input, output, input_names,output_names)
        
        self.ParameterInitialization()
        
        return None    

    def Simulation(self,x0,u,params=None,**kwargs):
        '''
        A iterative application of the OneStepPrediction in order to perform a
        simulation for a whole input trajectory
        x0: Casadi MX, inital state a begin of simulation
        u: Casadi MX,  input trajectory
        params: A dictionary of opti variables, if the parameters of the model
                should be optimized, if None, then the current parameters of
                the model are used
        '''
        if params==None:
            params = self.Parameters
        
        params_new = []
            
        for name in self.Function.name_in()[2::]:
 
            try:
                params_new.append(params[name])                      # Parameters are already in the right order as expected by Casadi Function
            except:
                params_new.append(self.Parameters[name])
        
        u = u[self.u_label].values
        
        # Rearrange input time series into time delayed input vectors
        u_delay = [u[i:i+self.dim_c,:].reshape((1,-1)) for i in range(u.shape[0]-self.dim_c+1)] 
        u_delay = np.vstack(u_delay)
        
        
        F_sim = self.Function.mapaccum(u_delay.shape[0])
        # print(params_new)
        x,y = F_sim(x0,u_delay.T,*params_new)
        
        x = x.T
        y = y.T

        return x,y 

class LinearSSM(recurrent):
    """
    
    """

    def __init__(self,dim_u,dim_x,dim_y,initial_params=None, 
                 frozen_params = [], init_proc='random', name='LinSSM'):
        
        self.dim_u = dim_u
        self.dim_x = dim_x
        self.dim_y = dim_y

        self.name = name
        
        self.initial_params = initial_params
        self.frozen_params = frozen_params
        self.init_proc = init_proc
        
        self.Initialize()

    def Initialize(self):
                    
        # For convenience of notation
        dim_u = self.dim_u
        dim_x = self.dim_x 
        dim_y = self.dim_y             
        name = self.name
        
        # Define input, state and output vector
        u = cs.MX.sym('u',dim_u,1)
        x = cs.MX.sym('x',dim_x,1)
        y = cs.MX.sym('y',dim_y,1)
        
        # Define Model Parameters
        A = cs.MX.sym('A',dim_x,dim_x)
        B = cs.MX.sym('B',dim_x,dim_u)
        C = cs.MX.sym('C',dim_y,dim_x)

        # Define Model Equations
        x_new = cs.mtimes(A,x) + cs.mtimes(B,u)
        y_new = cs.mtimes(C,x_new) 
        
        
        input = [x,u,A,B,C]
        input_names = ['x','u','A','B','C']
        
        output = [x_new,y_new]
        output_names = ['x_new','y_new']  
        
        self.Function = cs.Function(name, input, output, input_names,output_names)
        
        return None
   
class MLP():
    """
    Implementation of a single-layered Feedforward Neural Network.
    """

    def __init__(self,dim_u,dim_out,dim_hidden,u_label,y_label,name,initial_params=None, 
                 frozen_params = [], init_proc='random'):
        """
        Initialization procedure of the Feedforward Neural Network Architecture
        
        
        Parameters
        ----------
        dim_u : int
            Dimension of the input, e.g. dim_u = 2 if input is a 2x1 vector
        dim_out : int
            Dimension of the output, e.g. dim_out = 3 if output is a 3x1 vector.
        dim_hidden : int
            Number of nonlinear neurons in the hidden layer, e.g. dim_hidden=10,
            if NN is supposed to have 10 neurons in hidden layer.
        u_label : 
        name : str
            Name of the model, e.g. name = 'InjectionPhaseModel'.

        Returns
        -------
        None.

        """
        self.dim_u = dim_u
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        
        self.u_label = u_label
        self.y_label = y_label
        self.name = name
        
        self.initial_params = initial_params
        self.frozen_params = frozen_params
        self.init_proc = init_proc
        
        
        self.Initialize()

    def Initialize(self):
        """
        Defines the parameters of the model as symbolic casadi variables and 
        the model equation as casadi function. Model parameters are initialized
        randomly.

        Returns
        -------
        None.

        """   
        dim_u = self.dim_u
        dim_hidden = self.dim_hidden
        dim_out = self.dim_out 
        name = self.name
    
        u = cs.MX.sym('u',dim_u,1)
        x = cs.MX.sym('x',dim_out,1)
        
        # Model Parameters
        W_h = cs.MX.sym('W_h_'+name,dim_hidden,dim_u+dim_out)
        b_h = cs.MX.sym('b_h_'+name,dim_hidden,1)
        
        W_o = cs.MX.sym('W_out_'+name,dim_out,dim_hidden)
        b_o = cs.MX.sym('b_out_'+name,dim_out,1)

        # Model Equations
        h =  cs.tanh(cs.mtimes(W_h,cs.vertcat(u,x))+b_h)
        x_new = cs.mtimes(W_o,h)+b_o
        
        
        input = [x,u,W_h,b_h,W_o,b_o]
        input_names = [var.name() for var in input]
        
        output = [x_new]
        output_names = ['x_new']  
        
        self.Function = cs.Function(name, input, output, input_names,output_names)
        
        self.ParameterInitialization()
        
        return None
   
    def OneStepPrediction(self,x0,u0,params=None):
        """
        OneStepPrediction() evaluates the model equation defined in 
        self.Function()
        
        self.Function() takes initial state x0, input u0 and all model 
        parameters as input. The model parameters can either be optimization
        variables themselves (as in system identification) or the take specific 
        values (when the estimated model is used for control)

        Parameters
        ----------
        x0 : array-like with dimension [self.dim_x, 1]
            initial state resp. state from last time-step
        u0 : array-like with dimension [self.dim_u, 1]
            input
        params : dictionary, optional
            params is None: This is the case during model based control,
            self.Function() is evaluated with the numerical
            values of the model parameters saved in self.Parameters
            params is dictionary of opti.variables: During system identification
            the model parameters are optimization variables themselves, so a 
            dictionary of opti.variables is passed to self.Function()

        Returns
        -------
        x1 : array-like with dimension [self.dim_x, 1]
            output of the Feedforward Neural Network

        """
        if params==None:
            params = self.Parameters
        
        params_new = []
            
        for name in  self.Function.name_in():
            try:
                params_new.append(params[name])                      # Parameters are already in the right order as expected by Casadi Function
            except:
                continue
        
        x1 = self.Function(x0,u0,*params_new)     
                              
        return x1

    def Simulation(self,x0,u,params=None,**kwargs):
        '''
        A iterative application of the OneStepPrediction in order to perform a
        simulation for a whole input trajectory
        x0: Casadi MX, inital state a begin of simulation
        u: Casadi MX,  input trajectory
        params: A dictionary of opti variables, if the parameters of the model
                should be optimized, if None, then the current parameters of
                the model are used
        '''
        
        u = u[self.u_label].values
        
        x = []

        # initial states
        x.append(x0)
                      
        # Simulate Model
        for k in range(u.shape[0]):
            x_new  = self.OneStepPrediction(x[k],u[[k],:],params)
            x.append(x_new)
        
        # Concatenate list to casadiMX
        x = cs.hcat(x).T
       
        return x 

    def ParameterInitialization(self):
        '''
        Routine for parameter initialization. Takes input_names from the Casadi-
        Function defining the model equations self.Function and defines a 
        dictionary with input_names as keys. According to the initialization
        procedure defined in self.init_proc each key contains 
        a numpy array of appropriate shape

        Returns
        -------
        None.

        '''
                
        # Initialization procedure
        if self.init_proc == 'random':
            initialization = RandomInitialization
        elif self.init_proc == 'xavier':
            initialization = XavierInitialization
        elif self.init_proc == 'he':
            initialization = HeInitialization      
        
        # Define all parameters in a dictionary and initialize them 
        self.Parameters = {}
        
        new_param_values = {}
        for p_name in self.Function.name_in()[2::]:
            self.Parameters[p_name] = initialization(self.Function.size_in(p_name))
        
        # self.AssignParameters(new_param_values)

        # Initialize with specific inital parameters if given
        if self.initial_params is not None:
            for param in self.initial_params.keys():
                if param in self.Parameters.keys():
                    self.Parameters[param] = self.initial_params[param]
                    
    def SetParameters(self,params):
        for p_name in self.Function.name_in()[2::]:
            try:
                self.Parameters[p_name] = params[p_name]
            except:
                pass           

class Static_MLP(static):
    """
    Implementation of a single-layered Feedforward Neural Network.
    """

    def __init__(self,dim_u,dim_out,dim_hidden,u_label,y_label,name,
                 initial_params=None, frozen_params = [], init_proc='random'):
        """
        Initialization procedure of the Feedforward Neural Network Architecture
        
        
        Parameters
        ----------
        dim_u : int
            Dimension of the input, e.g. dim_u = 2 if input is a 2x1 vector
        dim_out : int
            Dimension of the output, e.g. dim_out = 3 if output is a 3x1 vector.
        dim_hidden : int
            Number of nonlinear neurons in the hidden layer, e.g. dim_hidden=10,
            if NN is supposed to have 10 neurons in hidden layer.
        name : str
            Name of the model, e.g. name = 'InjectionPhaseModel'.

        Returns
        -------
        None.

        """
        self.dim_u = dim_u
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        
        self.u_label = u_label
        self.y_label = y_label
        self.name = name
        
        self.initial_params = initial_params
        self.frozen_params = frozen_params
        self.init_proc = init_proc
        
        
        self.Initialize()

    def Initialize(self):
        """
        Defines the parameters of the model as symbolic casadi variables and 
        the model equation as casadi function. Model parameters are initialized
        randomly.

        Returns
        -------
        None.

        """   
        dim_u = self.dim_u
        dim_hidden = self.dim_hidden
        dim_out = self.dim_out 
        name = self.name
    
        u = cs.MX.sym('u',dim_u,1)
        
        # Model Parameters
        W_h = cs.MX.sym('W_h_'+name,dim_hidden,dim_u)
        b_h = cs.MX.sym('b_h_'+name,dim_hidden,1)
        
        W_o = cs.MX.sym('W_out_'+name,dim_out,dim_hidden)
        b_o = cs.MX.sym('b_out_'+name,dim_out,1)

        # Model Equations
        h =  cs.tanh(cs.mtimes(W_h,u)+b_h)
        y = cs.mtimes(W_o,h)+b_o
        
        
        input = [u,W_h,b_h,W_o,b_o]
        input_names = [var.name() for var in input]
        
        output = [y]
        output_names = ['y']  
        
        self.Function = cs.Function(name, input, output, input_names,output_names)
        
        self.ParameterInitialization()
        
        return None
   
class Static_Multi_MLP(static):
    """
    Implementation of a multi-layered Feedforward Neural Network.
    """

    def __init__(self,dim_u,dim_out,dim_hidden,layers,u_label,y_label,name,
                 initial_params=None, frozen_params = [], init_proc='random',
                 **kwargs):
        """
        Initialization procedure of the Feedforward Neural Network Architecture
        
        
        Parameters
        ----------
        dim_u : int
            Dimension of the input, e.g. dim_u = 2 if input is a 2x1 vector
        dim_out : int
            Dimension of the output, e.g. dim_out = 3 if output is a 3x1 vector.
        dim_hidden : int
            Number of nonlinear neurons in the hidden layer, e.g. dim_hidden=10,
            if NN is supposed to have 10 neurons in hidden layer.
        name : str
            Name of the model, e.g. name = 'InjectionPhaseModel'.

        Returns
        -------
        None.

        """
        self.dim_u = dim_u
        self.dim_hidden = dim_hidden
        self.layers = layers
        self.dim_out = dim_out
        
        self.u_label = u_label
        self.y_label = y_label
        self.name = name
        
        self.initial_params = initial_params
        self.frozen_params = frozen_params
        self.init_proc = init_proc
        
        
        self.Initialize()

    def Initialize(self):
        """
        Defines the parameters of the model as symbolic casadi variables and 
        the model equation as casadi function. Model parameters are initialized
        randomly.

        Returns
        -------
        None.

        """   
        dim_u = self.dim_u
        dim_hidden = self.dim_hidden
        layers = self.layers
        dim_out = self.dim_out 
        name = self.name
    
        u = cs.MX.sym('u',dim_u,1)
        
        # Define model parameters as CasADi symbolics
        
        # First hidden layer
        W_u = cs.MX.sym('W_u_'+name,dim_hidden,dim_u)
        b_u = cs.MX.sym('b_u_'+name,dim_hidden,1)
        
        # consecutive hidden layers
        W_h = cs.MX.sym('W_h_'+name + '_',dim_hidden,dim_hidden,layers-1)
        b_h = cs.MX.sym('b_h_'+name + '_',dim_hidden,1,layers-1)
        
        # Output layer
        W_o = cs.MX.sym('W_out_'+name,dim_out,dim_hidden)
        b_o = cs.MX.sym('b_out_'+name,dim_out,1)

        # Model Equations
        # Input to first hidden layer
        h =  cs.tanh(cs.mtimes(W_u,u)+b_u)
        
        # hidden layer to hidden layer
        for l in range(0,layers-1):
            h = cs.tanh(cs.mtimes(W_h[l],h)+b_h[l])
        
        # hidden layer to output layer
        y = cs.mtimes(W_o,h)+b_o
        
        input = [u,W_u,b_u,*W_h,*b_h,W_o,b_o]
        input_names = [var.name() for var in input]
                
        output = [y]
        output_names = ['y']  
        
        self.Function = cs.Function(name, input, output, input_names,output_names)
        
        self.ParameterInitialization()
        
        return None


class PolynomialModel(static):
    """
    Implementation of an n-th degree multivariate polynomial
    """

    def __init__(self,dim_u,dim_out,degree_n,interaction,u_label,y_label,name,
                 initial_params=None, frozen_params = [], init_proc='random'):
        """
        Initialization procedure of the Feedforward Neural Network Architecture
        
        
        Parameters
        ----------
        dim_u : int
            Dimension of the input, e.g. dim_u = 2 if input is a 2x1 vector
        dim_out : int
            Dimension of the output, e.g. dim_out = 3 if output is a 3x1 vector.
        degree_n : int
            Number of nonlinear neurons in the hidden layer, e.g. dim_hidden=10,
            if NN is supposed to have 10 neurons in hidden layer.
        interaction : bool
            Determines if interaction terms between inputs should exist (True) 
            or not (False)
        u_label : list
            List of strings containing the labels of the inputs, must be
            identical to columns in pandas dataframe given to the model
        y_label : list
            List of strings containing the labels of the outputs, must be
            identical to columns in pandas dataframe given to the model            
        name : str
            Name of the model, e.g. name = 'InjectionPhaseModel'.

        Returns
        -------
        None.

        """
        self.dim_u = dim_u
        self.degree_n = degree_n
        self.dim_out = dim_out
        
        self.u_label = u_label
        self.y_label = y_label
        self.name = name
        
        self.initial_params = initial_params
        self.frozen_params = frozen_params
        self.init_proc = init_proc
        
        
        self.Initialize()

    def Initialize(self):
        """
        Defines the parameters of the model as symbolic casadi variables and 
        the model equation as casadi function. Model parameters are initialized
        randomly.

        Returns
        -------
        None.

        """   
        dim_u = self.dim_u
        degree_n = self.degree_n
        dim_out = self.dim_out 
        name = self.name
    
        u = cs.MX.sym('u',dim_u,1)
        
        # Model Parameters
        w = cs.MX.sym('W_'+name,1,3)        # Replace dimensions 1 with appropriate dimensions of parameter


        # Model Equations
        y = cs.mtimes(w,u)                             # Replace with actual model equations
        
        
        input = [u,w]
        input_names = ['u','W_'+name]
        
        output = [y]
        output_names = ['y']  
        
        self.Function = cs.Function(name, input, output, input_names,output_names)
        
        self.ParameterInitialization()
        
        return None

   
def logistic(x):
    
    y = 0.5 + 0.5 * cs.tanh(0.5*x)

    return y

class GRU(recurrent):
    """
    Implementation of a Gated Recurrent Unit with a Feedforward Neural Network
    as output
    """

    def __init__(self,dim_u,dim_c,dim_hidden,u_label,y_label,dim_out,name,
                 initial_params={},frozen_params = [], init_proc='random'):
        """
        Initialization procedure of the GRU Architecture
        
        Parameters
        ----------
        dim_u : int
            Dimension of the input, e.g. dim_u = 2 if input is a 2x1 vector
        dim_c : int
            Dimension of the cell-state, i.e. the internal state of the GRU,
            e.g. dim_c = 2 if cell-state is a 2x1 vector
        dim_hidden : int
            Number of nonlinear neurons in the hidden layer, e.g. dim_hidden=10,
            if output network is supposed to have 10 neurons in hidden layer.           
        dim_out : int
            Dimension of the output, e.g. dim_out = 3 if output is a 3x1 vector.
        name : str
            Name of the model, e.g. name = 'QualityModel'.

        Returns
        -------
        None.

        """        
        self.dim_u = dim_u
        self.dim_c = dim_c
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        
        self.u_label = u_label
        self.y_label = y_label
        self.name = name
        
        self.initial_params = initial_params
        self.frozen_params = frozen_params
        self.init_proc = init_proc
        
        self.dynamics = 'internal'
        
        self.Initialize()  
 

    def Initialize(self):
        """
        Defines the parameters of the model as symbolic casadi variables and 
        the model equation as casadi function. Model parameters are initialized
        randomly.

        Returns
        -------
        None.

        """          
        dim_u = self.dim_u
        dim_c = self.dim_c
        dim_hidden = self.dim_hidden
        dim_out = self.dim_out
        name = self.name      
        
        u = cs.MX.sym('u',dim_u,1)
        c = cs.MX.sym('c',dim_c,1)
 
 
        # Parameters
        # RNN part
        W_r = cs.MX.sym('W_r_'+name,dim_c,dim_u+dim_c)
        b_r = cs.MX.sym('b_r_'+name,dim_c,1)
    
        W_z = cs.MX.sym('W_z_'+name,dim_c,dim_u+dim_c)
        b_z = cs.MX.sym('b_z_'+name,dim_c,1)    
        
        W_c = cs.MX.sym('W_c_'+name,dim_c,dim_u+dim_c)
        b_c = cs.MX.sym('b_c_'+name,dim_c,1)    
    
        # MLP part
        W_h = cs.MX.sym('W_h_'+name,dim_hidden,dim_c)
        b_h = cs.MX.sym('b_h_'+name,dim_hidden,1)    
        
        W_y = cs.MX.sym('W_y_'+name,dim_out,dim_hidden)
        b_y = cs.MX.sym('b_y_'+name,dim_out,1)  
        
        
        # Equations
        f_r = logistic(cs.mtimes(W_r,cs.vertcat(u,c))+b_r)
        f_z = logistic(cs.mtimes(W_z,cs.vertcat(u,c))+b_z)
        
        c_r = f_r*c
        
        f_c = cs.tanh(cs.mtimes(W_c,cs.vertcat(u,c_r))+b_c)
        
        
        c_new = f_z*f_c+(1-f_z)*c
        
        h =  cs.tanh(cs.mtimes(W_h,c_new)+b_h)
        x_new = cs.mtimes(W_y,h)+b_y    
    
        
        # Casadi Function
        input = [c,u,W_r,b_r,W_z,b_z,W_c,b_c,W_h,b_h,W_y,b_y]
        input_names = [var.name() for var in input]
        
        output = [c_new,x_new]
        output_names = ['c_new','x_new']
    
        self.Function = cs.Function(name, input, output, input_names,output_names)
        
        self.ParameterInitialization()

        return None
    
class LSTM(recurrent):
    """
    Implementation of a LSTM Unit with a Feedforward Neural Network
    as output
    """

    def __init__(self,dim_u,dim_c,dim_hidden,dim_out,name,initial_params=None, 
                 frozen_params = [], init_proc='random'):
        """
        Initialization procedure of the LSTM Architecture
        
        Parameters
        ----------
        dim_u : int
            Dimension of the input, e.g. dim_u = 2 if input is a 2x1 vector
        dim_c : int
            Dimension of the cell-state, i.e. the internal state of the GRU,
            e.g. dim_c = 2 if cell-state is a 2x1 vector
        dim_hidden : int
            Number of nonlinear neurons in the hidden layer, e.g. dim_hidden=10,
            if output network is supposed to have 10 neurons in hidden layer.           
        dim_out : int
            Dimension of the output, e.g. dim_out = 3 if output is a 3x1 vector.
        name : str
            Name of the model, e.g. name = 'QualityModel'.

        Returns
        -------
        None.

        """        
        self.dim_u = dim_u
        self.dim_c = dim_c
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.name = name
        
        self.initial_params = initial_params
        self.frozen_params = frozen_params
        self.init_proc = init_proc
        
        self.Initialize()  
 

    def Initialize(self):
        """
        Defines the parameters of the model as symbolic casadi variables and 
        the model equation as casadi function. Model parameters are initialized
        randomly.

        Returns
        -------
        None.

        """          
        dim_u = self.dim_u
        dim_c = self.dim_c
        dim_hidden = self.dim_hidden
        dim_out = self.dim_out
        name = self.name      
        
        u = cs.MX.sym('u',dim_u,1)
        c = cs.MX.sym('c',dim_c,1)
        h = cs.MX.sym('h',dim_c,1)
        
        # Parameters
        # RNN part
        W_f = cs.MX.sym('W_f_'+name,dim_c,dim_u+dim_c)
        b_f = cs.MX.sym('b_f_'+name,dim_c,1)
    
        W_i = cs.MX.sym('W_i_'+name,dim_c,dim_u+dim_c)
        b_i = cs.MX.sym('b_i_'+name,dim_c,1)    
        
        W_o = cs.MX.sym('W_o_'+name,dim_c,dim_u+dim_c)
        b_o = cs.MX.sym('b_o_'+name,dim_c,1)    

        W_c = cs.MX.sym('W_c_'+name,dim_c,dim_u+dim_c)
        b_c = cs.MX.sym('b_c_'+name,dim_c,1)     
    
        # MLP part
        W_h = cs.MX.sym('W_h_'+name,dim_hidden,dim_c)
        b_h = cs.MX.sym('b_h_'+name,dim_hidden,1)    
        
        W_y = cs.MX.sym('W_y_'+name,dim_out,dim_hidden)
        b_y = cs.MX.sym('b_y_'+name,dim_out,1)  
        
        
        # Equations
        # RNN part
        f_f = logistic(cs.mtimes(W_f,cs.vertcat(u,h))+b_f)
        f_i = logistic(cs.mtimes(W_i,cs.vertcat(u,h))+b_i)
        f_o = logistic(cs.mtimes(W_o,cs.vertcat(u,h))+b_o)
        f_c = cs.tanh(cs.mtimes(W_c,cs.vertcat(u,h))+b_c)
        
        c_new = f_f*c + f_i*f_c
        h_new = f_o * cs.tanh(c_new)
        
        
        # MLP part
                
        MLP_h =  cs.tanh(cs.mtimes(W_h,h_new)+b_h)
        y_new = cs.mtimes(W_y,MLP_h)+b_y    
    
        
        # Casadi Function
        input = [c,h,u,W_f,b_f,W_i,b_i,W_o,b_o,W_c,b_c,W_h,b_h,W_y,b_y]
        input_names = [var.name() for var in input]
        
        output = [c_new, h_new, y_new]
        output_names = ['c_new','h_new','y_new']
    
        self.Function = cs.Function(name, input, output, input_names,output_names)
        
        self.ParameterInitialization()

        return None

    def OneStepPrediction(self,c0,h0,u0,params=None):
        '''
        Estimates the next state and output from current state and input
        x0: Casadi MX, current state
        u0: Casadi MX, current input
        params: A dictionary of opti variables, if the parameters of the model
                should be optimized, if None, then the current parameters of
                the model are used
        '''
        
        if params==None:
            params = self.Parameters
        
        params_new = []
            
        for name in self.Function.name_in()[3::]:
 
            try:
                params_new.append(params[name])                      # Parameters are already in the right order as expected by Casadi Function
            except:
                params_new.append(self.Parameters[name])  
            
        c1,h1,y1 = self.Function(c0,h0,u0,*params_new)     
                              
        return c1,h1,y1
   
    def Simulation(self,x0,u,params=None):
        '''
        A iterative application of the OneStepPrediction in order to perform a
        simulation for a whole input trajectory
        x0: Casadi MX, inital state a begin of simulation
        u: Casadi MX,  input trajectory
        params: A dictionary of opti variables, if the parameters of the model
                should be optimized, if None, then the current parameters of
                the model are used
        '''

        x = []
        y = []

        # initial states
        x.append(x0)
        h_old = x0
               
        # Simulate Model
        for k in range(u.shape[0]):
            c_new,h_new,y_new = self.OneStepPrediction(x[k],h_old,u[[k],:],params)
            x.append(c_new)
            h_old=h_new
            y.append(y_new)
        
        # Concatenate list to casadiMX
        y = cs.hcat(y).T    
        x = cs.hcat(x).T
       
        return x,y    
    
    def ParameterInitialization(self):
        '''
        Routine for parameter initialization. Takes input_names from the Casadi-
        Function defining the model equations self.Function and defines a 
        dictionary with input_names as keys. According to the initialization
        procedure defined in self.init_proc each key contains 
        a numpy array of appropriate shape

        Returns
        -------
        None.

        '''
                
        # Initialization procedure
        if self.init_proc == 'random':
            initialization = RandomInitialization
        elif self.init_proc == 'xavier':
            initialization = XavierInitialization
        elif self.init_proc == 'he':
            initialization = HeInitialization      
        
        # Define all parameters in a dictionary and initialize them 
        self.Parameters = {}
        
        new_param_values = {}
        for p_name in self.Function.name_in()[3::]:
            new_param_values[p_name] = initialization(self.Function.size_in(p_name))
        
        self.AssignParameters(new_param_values)

        # Initialize with specific inital parameters if given
        if self.initial_params is not None:
            for param in self.initial_params.keys():
                if param in self.Parameters.keys():
                    self.Parameters[param] = self.initial_params[param]
                    
    def AssignParameters(self,params):
        
        for p_name in self.Function.name_in()[3::]:
            self.Parameters[p_name] = params[p_name]
    

class LSS(recurrent):
    """
    Implementation of a linear state space model with a nonlinear output layer
    """

    def __init__(self,dim_u,dim_c,dim_hidden,dim_out,name,initial_params={}, 
                 frozen_params = [], init_proc='random',A_eig=[]):
        """
        Initialization procedure of the GRU Architecture
        
        Parameters
        ----------
        dim_u : int
            Dimension of the input, e.g. dim_u = 2 if input is a 2x1 vector
        dim_c : int
            Dimension of the cell-state, i.e. the internal state of the GRU,
            e.g. dim_c = 2 if cell-state is a 2x1 vector
        dim_hidden : int
            Number of nonlinear neurons in the hidden layer, e.g. dim_hidden=10,
            if output network is supposed to have 10 neurons in hidden layer.           
        dim_out : int
            Dimension of the output, e.g. dim_out = 3 if output is a 3x1 vector.
        name : str
            Name of the model, e.g. name = 'QualityModel'.

        Returns
        -------
        None.

        """        
        self.dim_u = dim_u
        self.dim_c = dim_c
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.A_eig = A_eig
        self.name = name
        
        self.initial_params = initial_params
        self.frozen_params = frozen_params
        self.init_proc = init_proc
        
        self.Initialize()  
 

    def Initialize(self):
        """
        Defines the parameters of the model as symbolic casadi variables and 
        the model equation as casadi function. Model parameters are initialized
        randomly.

        Returns
        -------
        None.

        """          
        dim_u = self.dim_u
        dim_c = self.dim_c
        dim_hidden = self.dim_hidden
        dim_out = self.dim_out
        name = self.name      
        
        u = cs.MX.sym('u',dim_u,1)
        c = cs.MX.sym('c',dim_c,1)
 
 
        # Parameters
        # Recurrent part
        A = cs.MX.sym('A_r_'+name,dim_c,dim_c)
        B = cs.MX.sym('B_z_'+name,dim_c,dim_u)

    
        # MLP part
        W_h = cs.MX.sym('W_h_'+name,dim_hidden,dim_c)
        b_h = cs.MX.sym('b_h_'+name,dim_hidden,1)    
        
        W_y = cs.MX.sym('W_y_'+name,dim_out,dim_hidden)
        b_y = cs.MX.sym('b_y_'+name,dim_out,1)  
        
        
        # Equations
        c_new = cs.mtimes(A,c) + cs.mtimes(B,u)
        h_new = cs.tanh(cs.mtimes(W_h,c_new)+b_h)
        y_new = cs.mtimes(W_y,h_new)+b_y    
    
        
        # Casadi Function
        input = [c,u,A,B,W_h,b_h,W_y,b_y]
        input_names = [var.name() for var in input]
        
        output = [c_new,y_new]
        output_names = ['c_new','y_new']
    
        self.Function = cs.Function(name, input, output, input_names,output_names)
        
        self.ParameterInitialization()

        return None

    def ParameterInitialization(self):
        '''
        Routine for parameter initialization. Takes input_names from the Casadi-
        Function defining the model equations self.Function and defines a 
        dictionary with input_names as keys. According to the initialization
        procedure defined in self.init_proc each key contains 
        a numpy array of appropriate shape

        Returns
        -------
        None.

        '''
                
        # Initialization procedure
        if self.init_proc == 'random':
            initialization = RandomInitialization
        elif self.init_proc == 'xavier':
            initialization = XavierInitialization
        elif self.init_proc == 'he':
            initialization = HeInitialization      
        
        # Define all parameters in a dictionary and initialize them 
        self.Parameters = {}
        
        new_param_values = {}
        for p_name in self.Function.name_in()[2::]:
            self.Parameters[p_name] = initialization(self.Function.size_in(p_name))
        
        # Initialize A-matrix such that resulting system is stable
        A_key = self.Function.name_in()[2]
        
        if len(self.A_eig) == 0:
            self.A_eig =  np.random.uniform(-1,1,(self.dim_c))
        
        if self.dim_c>1:
            Q = ortho_group.rvs(dim=self.dim_c)
            self.Parameters[A_key] = Q.T.dot(np.diag(self.A_eig).dot(Q))
        elif self.dim_c==1:
            self.Parameters[A_key] = self.A_eig.reshape((self.dim_c,self.dim_c))
        
        # Initialize with specific inital parameters if given
        if self.initial_params is not None:
            for param in self.initial_params.keys():
                if param in self.Parameters.keys():
                    self.Parameters[param] = self.initial_params[param]
        
        return None