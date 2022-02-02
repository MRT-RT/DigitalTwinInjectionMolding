# -*- coding: utf-8 -*-

# from sys import path
# path.append(r"C:\Users\LocalAdmin\Documents\casadi-windows-py38-v3.5.5-64bit")
# import os
# print (os.getcwd())
import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from DIM.optim.common import RK4
from .initializations import XavierInitialization, RandomInitialization, HeInitialization

# from miscellaneous import *

class RNN():
    '''
    Parent class for all Models
    '''

    def ParameterInitialization(self):
        '''
        Routine for parameter initialization. Takes input_names from the Casadi-
        Function defining the model equations self.Function and defines a 
        dictionary with input_names as keys. According to the initialization
        procedure defined in self.InitializationProcedure each key contains 
        a numpy array of appropriate shape

        Returns
        -------
        None.

        '''
                
        # Initialization procedure
        if self.InitializationProcedure == 'random':
            initialization = RandomInitialization
        elif self.InitializationProcedure == 'xavier':
            initialization = XavierInitialization
        elif self.InitializationProcedure == 'he':
            initialization = HeInitialization      
        
        # Define all parameters in a dictionary and initialize them 
        self.Parameters = {}
        
        new_param_values = {}
        for p_name in self.Function.name_in()[2::]:
            new_param_values[p_name] = initialization(self.Function.size_in(p_name))
        
        self.AssignParameters(new_param_values)

        # Initialize with specific inital parameters if given
        if self.InitialParameters is not None:
            for param in self.InitialParameters.keys():
                if param in self.Parameters.keys():
                    self.Parameters[param] = self.InitialParameters[param]
                    
    def AssignParameters(self,params):
        
        for p_name in self.Function.name_in()[2::]:
            self.Parameters[p_name] = params[p_name]
                        

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
                params_new.append(self.Parameters[name])  
            
        x1,y1 = self.Function(x0,u0,*params_new)     
                              
        return x1,y1
   
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
                      
        # Simulate Model
        for k in range(u.shape[0]):
            x_new,y_new = self.OneStepPrediction(x[k],u[[k],:],params)
            x.append(x_new)
            y.append(y_new)
        
        # Concatenate list to casadiMX
        y = cs.hcat(y).T    
        x = cs.hcat(x).T
       
        return x,y    

class LinearSSM(RNN):
    """
    
    """

    def __init__(self,dim_u,dim_x,dim_y,initial_params=None, 
                 frozen_params = [], init_proc='random', name='LinSSM'):
        
        self.dim_u = dim_u
        self.dim_x = dim_x
        self.dim_y = dim_y

        self.name = name
        
        self.InitialParameters = initial_params
        self.FrozenParameters = frozen_params
        self.InitializationProcedure = init_proc
        
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

    def __init__(self,dim_u,dim_out,dim_hidden,initial_params=None, 
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
        name : str
            Name of the model, e.g. name = 'InjectionPhaseModel'.

        Returns
        -------
        None.

        """
        self.dim_u = dim_u
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.name = name
        
        self.InitialParameters = initial_params
        self.FrozenParameters = frozen_params
        self.InitializationProcedure = init_proc
        
        
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
        input_names = ['x','u','W_h_'+name,'b_h_'+name,'W_o_'+name,'b_o_'+name]
        
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
   

    def ParameterInitialization(self):
        '''
        Routine for parameter initialization. Takes input_names from the Casadi-
        Function defining the model equations self.Function and defines a 
        dictionary with input_names as keys. According to the initialization
        procedure defined in self.InitializationProcedure each key contains 
        a numpy array of appropriate shape

        Returns
        -------
        None.

        '''
                
        # Initialization procedure
        if self.InitializationProcedure == 'random':
            initialization = RandomInitialization
        elif self.InitializationProcedure == 'xavier':
            initialization = XavierInitialization
        elif self.InitializationProcedure == 'he':
            initialization = HeInitialization      
        
        # Define all parameters in a dictionary and initialize them 
        self.Parameters = {}
        
        new_param_values = {}
        for p_name in self.Function.name_in()[2::]:
            new_param_values[p_name] = initialization(self.Function.size_in(p_name))
        
        self.AssignParameters(new_param_values)

        # Initialize with specific inital parameters if given
        if self.InitialParameters is not None:
            for param in self.InitialParameters.keys():
                if param in self.Parameters.keys():
                    self.Parameters[param] = self.InitialParameters[param]
                    
    def AssignParameters(self,params):
        
        for p_name in self.Function.name_in()[2::]:
            self.Parameters[p_name] = params[p_name]

class Static_MLP():
    """
    Implementation of a single-layered Feedforward Neural Network.
    """

    def __init__(self,dim_u,dim_out,dim_hidden,name,initial_params=None, 
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
        name : str
            Name of the model, e.g. name = 'InjectionPhaseModel'.

        Returns
        -------
        None.

        """
        self.dim_u = dim_u
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.name = name
        
        self.InitialParameters = initial_params
        self.FrozenParameters = frozen_params
        self.InitializationProcedure = init_proc
        
        
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
        input_names = ['u','W_h_'+name,'b_h_'+name,'W_o_'+name,'b_o_'+name]
        
        output = [y]
        output_names = ['y']  
        
        self.Function = cs.Function(name, input, output, input_names,output_names)
        
        self.ParameterInitialization()
        
        return None
   
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
   

    def ParameterInitialization(self):
        '''
        Routine for parameter initialization. Takes input_names from the Casadi-
        Function defining the model equations self.Function and defines a 
        dictionary with input_names as keys. According to the initialization
        procedure defined in self.InitializationProcedure each key contains 
        a numpy array of appropriate shape

        Returns
        -------
        None.

        '''
                
        # Initialization procedure
        if self.InitializationProcedure == 'random':
            initialization = RandomInitialization
        elif self.InitializationProcedure == 'xavier':
            initialization = XavierInitialization
        elif self.InitializationProcedure == 'he':
            initialization = HeInitialization      
        
        # Define all parameters in a dictionary and initialize them 
        self.Parameters = {}
        
        new_param_values = {}
        for p_name in self.Function.name_in()[1::]:
            new_param_values[p_name] = initialization(self.Function.size_in(p_name))
        
        self.AssignParameters(new_param_values)

        # Initialize with specific inital parameters if given
        if self.InitialParameters is not None:
            for param in self.InitialParameters.keys():
                if param in self.Parameters.keys():
                    self.Parameters[param] = self.InitialParameters[param]
                    
    def AssignParameters(self,params):
        
        for p_name in self.Function.name_in()[1::]:
            self.Parameters[p_name] = params[p_name]

    
def logistic(x):
    
    y = 0.5 + 0.5 * cs.tanh(0.5*x)

    return y

class GRU(RNN):
    """
    Implementation of a Gated Recurrent Unit with a Feedforward Neural Network
    as output
    """

    def __init__(self,dim_u,dim_c,dim_hidden,dim_out,name,initial_params=None, 
                 frozen_params = [], init_proc='random'):
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
        self.name = name
        
        self.InitialParameters = initial_params
        self.FrozenParameters = frozen_params
        self.InitializationProcedure = init_proc
        
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
        input_names = ['c','u','W_r_'+name,'b_r_'+name,'W_z_'+name,'b_z_'+name
                       ,'W_c_'+name,'b_c_'+name,'W_h_'+name,'b_h_'+name,
                        'W_y_'+name,'b_y_'+name]
        
        output = [c_new,x_new]
        output_names = ['c_new','x_new']
    
        self.Function = cs.Function(name, input, output, input_names,output_names)
        
        self.ParameterInitialization()

        return None
    
class LSTM(RNN):
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
        
        self.InitialParameters = initial_params
        self.FrozenParameters = frozen_params
        self.InitializationProcedure = init_proc
        
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
        input_names = ['c','h','u','W_f_'+name,'b_f_'+name,'W_i_'+name,
                       'b_i_'+name,'W_o_'+name,'b_o_'+name,'W_c_'+name,
                       'b_c_'+name,'W_h_'+name,'b_h_'+name,'W_y_'+name,
                       'b_y_'+name]
        
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
        procedure defined in self.InitializationProcedure each key contains 
        a numpy array of appropriate shape

        Returns
        -------
        None.

        '''
                
        # Initialization procedure
        if self.InitializationProcedure == 'random':
            initialization = RandomInitialization
        elif self.InitializationProcedure == 'xavier':
            initialization = XavierInitialization
        elif self.InitializationProcedure == 'he':
            initialization = HeInitialization      
        
        # Define all parameters in a dictionary and initialize them 
        self.Parameters = {}
        
        new_param_values = {}
        for p_name in self.Function.name_in()[3::]:
            new_param_values[p_name] = initialization(self.Function.size_in(p_name))
        
        self.AssignParameters(new_param_values)

        # Initialize with specific inital parameters if given
        if self.InitialParameters is not None:
            for param in self.InitialParameters.keys():
                if param in self.Parameters.keys():
                    self.Parameters[param] = self.InitialParameters[param]
                    
    def AssignParameters(self,params):
        
        for p_name in self.Function.name_in()[3::]:
            self.Parameters[p_name] = params[p_name]
    
    
class FirstOrderSystem():
    """
    
    """

    def __init__(self,dt,name):
        
        self.name = name
        self.dt = dt
        
        self.dim_u = 1
        self.dim_out = 1
        
        self.Initialize()

    def Initialize(self):
            
            # For convenience of notation
            name = self.name
            dt = self.dt
            dim_u = self.dim_u
            dim_out = self.dim_out
            
            
            # Define input, state and output vector
            u = cs.MX.sym('u',1,1)
            x = cs.MX.sym('x',1,1)
            
            # Define Model Parameters
            a = cs.MX.sym('a',1,1)
            b = cs.MX.sym('b',1,1)
            
            # Put all Parameters in Dictionary with random initialization
            self.Parameters = {'a':np.random.rand(1,1),
                               'b':np.random.rand(1,1)}
        
            # continuous dynamics
            x_new = a*x + b*u

            input = [x,u,a,b]
            input_names = ['x','u','a','b']
            
            output = [x_new]
            output_names = ['x_new']  

            f_cont = cs.Function(name,input,output,
                                 input_names,output_names)  
            
            x1 = RK4(f_cont,input,dt)
          
            
            self.Function = cs.Function(name,input,[x1],input_names,output_names)
            
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
            
        for name in  self.Function.name_in():
            try:
                params_new.append(params[name])                      # Parameters are already in the right order as expected by Casadi Function
            except:
                continue
        
        x1 = self.Function(x0,u0,*params_new)     
                              
        return x1
   
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

        # initial states
        x.append(x0)
                      
        # Simulate Model
        for k in range(u.shape[0]):
            x_new = self.OneStepPrediction(x[k],u[[k],:],params)
            x.append(x_new)
        

        # Concatenate list to casadiMX
  
        x = cs.hcat(x).T

       
        return x
    
class SecondOrderSystem():
    """
    
    """

    def __init__(self,dt,name):
        
        self.name = name
        self.dt = dt
        
        self.dim_u = 1
        self.dim_out = 1
        
        self.Initialize()

    def Initialize(self):
            
            # For convenience of notation
            name = self.name
            dt = self.dt
            dim_u = self.dim_u
            dim_out = self.dim_out
            
            # Define input, state and output vector
            u = cs.MX.sym('u',dim_u,1)
            x = cs.MX.sym('x',2,1)
            
            # Define Model Parameters
            A = cs.MX.sym('A',2,2)
            b = cs.MX.sym('b',2,1)
            c = cs.MX.sym('c',1,2)
            
            # Put all Parameters in Dictionary with random initialization
            self.Parameters = {'A':np.random.rand(2,2),
                               'b':np.random.rand(2,dim_u),
                               'c':np.random.rand(dim_out,2)}
        
            # continuous dynamics
            x_new = cs.mtimes(A,x) + cs.mtimes(b,u)

            input = [x,u,A,b,c]
            input_names = ['x','u','A','b','c']
            
            output = [x_new]
            output_names = ['x_new']  

            f_cont = cs.Function(name,input,output,
                                 input_names,output_names)  
            
            x1 = RK4(f_cont,input,dt)
            y1 = cs.mtimes(c,x1)
            
            self.Function = cs.Function(name,input,[x1,y1],input_names,['x_new','y_new'])
            
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
            
        for name in  self.Function.name_in():
            try:
                params_new.append(params[name])                      # Parameters are already in the right order as expected by Casadi Function
            except:
                continue
        
        x1,y1 = self.Function(x0,u0,*params_new)     
                              
        return x1,y1
   
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
                      
        # Simulate Model
        for k in range(u.shape[0]):
            x_new,y_new = self.OneStepPrediction(x[k],u[[k],:],params)
            x.append(x_new)
            y.append(y_new)

        # Concatenate list to casadiMX
  
        x = cs.hcat(x).T
        y = cs.hcat(y).T
       
        return x,y