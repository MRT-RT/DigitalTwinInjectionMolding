# -*- coding: utf-8 -*-

from sys import path
path.append(r"C:\Users\LocalAdmin\Documents\casadi-windows-py38-v3.5.5-64bit")

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np

# from miscellaneous import *


class ProcessModel():
    """
    Modell der Spritzgießmaschine, welches Führungsgrößen (parametriert durch 
    an der Maschine einstellbare Größen) auf die resultierenden Prozessgrößen
    abbildet.    
    """

    def __init__(self):
        
        self.reference = []
        self.ref_params = {}       
        self.subsystems = []
        self.switching_instances = []
        
        
class QualityModel():
    '''
    Container for the model which estimates the quality of the part given 
    trajectories of the process variables
    '''
    def __init__(self,subsystems,name):
        '''
        Initialization routine for the QualityModel 
        model: model_structures, needs to be a recurrent model which can map
               trajectories to a single final value
        u0: int, length of the cycle in discrete time steps 
        '''
       
        self.subsystems = subsystems
        self.switching_instances = []
        self.name = name
        # self.FrozenParameters = frozen_params
        
        dim_c = []
        dim_out = []
        
        for subsystem in self.subsystems:
            dim_c.append(subsystem.dim_c)
            dim_out.append(subsystem.dim_out)
            
            object.__setattr__(self, 'dim_u'+'_'+subsystem.name, 
                               subsystem.dim_u)
            object.__setattr__(self, 'dim_hidden'+'_'+subsystem.name, 
                               subsystem.dim_hidden)
        
        # Check consistency
        if sum(dim_c)/len(dim_c)==dim_c[0]:
            self.dim_c = dim_c[0]
        else:
            raise ValueError('Cell state of all subsystems needs to be equal')
        
        if sum(dim_out)/len(dim_out)==dim_out[0]:
            self.dim_out = dim_out[0]
        else:
            raise ValueError('Cell state of all subsystems needs to be equal')
            
        self.Initialize()
        
        
        
    def Initialize(self):
        
        '''
        Write a routine that initializes subsystem models automatically, needed
        for PSO algorithm
        '''
       
        # Update attributes of each subsystem
        for subsystem in self.subsystems:
            
            setattr(subsystem, 'dim_u', 
                    object.__getattribute__(self,'dim_u'+'_'+subsystem.name))
            setattr(subsystem, 'dim_c', 
                    object.__getattribute__(self,'dim_c'))
            setattr(subsystem, 'dim_hidden', 
                    object.__getattribute__(self,'dim_hidden'+'_'+subsystem.name))
            setattr(subsystem, 'dim_out',
                    object.__getattribute__(self,'dim_out'))
            
            # Call Initialize function of each subsystem
            subsystem.Initialize()
                
        return None
    
    def Simulation(self,c0,u,params=None,switching_instances=None):
        """
        Simulates the quality model for a given input trajectory u and an initial
        hidden state (cell state of RNN) 
    
        Parameters
        ----------
        c0 : array-like,
             Initial hidden state (cell state), i.e. the internal state of the 
             GRU or LSTM, e.g. if dim_c = 2 then c0 is a 2x1 vector
        u : array-like with dimension [N,self.dim_u]
            trajectory of input signal, i.e. a vector with dimension N x dim_u
    
        Returns
        -------
        c : array-like,
            Vector containing trajectory of simulated hidden cell state, e.g.
            for a simulation over N time steps and dim_c = 2 c is a Nx2 vector
        y : array-like,
            Vector containing trajectory of simulated output, e.g. for
            a simulation over N time steps and dim_out = 3 y is a Nx3 vector
    
        """
        self.switching_instances = switching_instances
        
        # Create empty arrays for output y and hidden state c
        y = []
        c = []    
        
        # Initial hidden state
        # X.append(np.zeros((c_dims[0],1)))
        
        # Intervalls to divide input according to switching instances

        # add zero and len(u) as indices
        # ind = [0] + self.switching_instances + [u.shape[0]]  
          
                  
        # intervals = [[ind[k],ind[k+1]] for k in range(0,len(ind)-1)]
        
        # c0 = np.zeros((self.dim_c,1))
        
        

        # System Dynamics as Path Constraints
        for system,u_sys in zip(self.subsystems,u):
            
            # Do a one step prediction based on the model
            sim = system.Simulation(c0,u_sys,params)
            
            # Last hidden state is inital state for next model
            c0 = sim[0][-1,:].T
            
            # OneStepPrediction can return a tuple, i.e. state and output. The 
            # output is per convention the second entry
            if isinstance(sim,tuple):
                c.append(sim[0])
                y.append(sim[1])        
            
        # Concatenate list to casadiMX
        y = cs.vcat(y)  
        c = cs.vcat(c)          
            
        return c,y  
    
    def ParameterInitialization(self):
        
        self.Parameters = {}
        self.FrozenParameters = []
        
        for system in self.subsystems:
            system.ParameterInitialization()
            self.Parameters.update(system.Parameters)                                  # append subsystems parameters
            self.FrozenParameters.extend(system.FrozenParameters)

    def AssignParameters(self,params):
        self.Parameters = params
        
        for system in self.subsystems:
            system.AssignParameters(params)
                

class LinearSSM():
    """
    
    """

    def __init__(self,dim_u,dim_x,dim_y,name):
        
        self.dim_u = dim_u
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.name = name
        
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

            
            # Put all Parameters in Dictionary with random initialization
            self.Parameters = {'A':np.random.rand(dim_x,dim_x),
                               'B':np.random.rand(dim_x,dim_u),
                               'C':np.random.rand(dim_y,dim_x)}
        
            # self.Input = {'u':np.random.rand(u.shape)}
            
            # Define Model Equations
            x_new = cs.mtimes(A,x) + cs.mtimes(B,u)
            y_new = cs.mtimes(C,x_new) 
            
            
            input = [x,u,A,B,C]
            input_names = ['x','u','A','B','C']
            
            output = [x_new,y_new]
            output_names = ['x_new','y_new']  
            
            self.Function = cs.Function(name, input, output, input_names,output_names)
            
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
        y = cs.hcat(y).T    
        x = cs.hcat(x).T
       
        return y


# class MLP():
#     """
#     Implementation of a single-layered Feedforward Neural Network.
#     """

#     def __init__(self,dim_u,dim_x,dim_hidden,name):
#         """
#         Initialization procedure of the Feedforward Neural Network Architecture
        
        
#         Parameters
#         ----------
#         dim_u : int
#             Dimension of the input, e.g. dim_u = 2 if input is a 2x1 vector
#         dim_x : int
#             Dimension of the state, e.g. dim_x = 3 if state is a 3x1 vector.
#         dim_hidden : int
#             Number of nonlinear neurons in the hidden layer, e.g. dim_hidden=10,
#             if NN is supposed to have 10 neurons in hidden layer.
#         name : str
#             Name of the model, e.g. name = 'InjectionPhaseModel'.

#         Returns
#         -------
#         None.

#         """
#         self.dim_u = dim_u
#         self.dim_hidden = dim_hidden
#         self.dim_x = dim_x
#         self.name = name
        
#         self.Initialize()

#     def Initialize(self):
#         """
#         Defines the parameters of the model as symbolic casadi variables and 
#         the model equation as casadi function. Model parameters are initialized
#         randomly.

#         Returns
#         -------
#         None.

#         """   
#         dim_u = self.dim_u
#         dim_hidden = self.dim_hidden
#         dim_x = self.dim_x 
#         name = self.name
    
#         u = cs.MX.sym('u',dim_u,1)
#         x = cs.MX.sym('x',dim_x,1)
        
#         # Model Parameters
#         W_h = cs.MX.sym('W_h',dim_hidden,dim_u+dim_x)
#         b_h = cs.MX.sym('b_h',dim_hidden,1)
        
#         W_o = cs.MX.sym('W_out',dim_x,dim_hidden)
#         b_o = cs.MX.sym('b_out',dim_x,1)
        
#         # Put all Parameters in Dictionary with random initialization
#         self.Parameters = {'W_h':np.random.rand(W_h.shape[0],W_h.shape[1]),
#                            'b_h':np.random.rand(b_h.shape[0],b_h.shape[1]),
#                            'W_o':np.random.rand(W_o.shape[0],W_o.shape[1]),
#                            'b_o':np.random.rand(b_o.shape[0],b_o.shape[1])}
    
       
#         # Model Equations
#         h =  cs.tanh(cs.mtimes(W_h,cs.vertcat(u,x))+b_h)
#         x_new = cs.mtimes(W_o,h)+b_o
        
        
#         input = [x,u,W_h,b_h,W_o,b_o]
#         input_names = ['x','u','W_h','b_h','W_o','b_o']
        
#         output = [x_new]
#         output_names = ['x_new']  
        
#         self.Function = cs.Function(name, input, output, input_names,output_names)
        
#         return None
   
#     def OneStepPrediction(self,x0,u0,params=None):
#         """
#         OneStepPrediction() evaluates the model equation defined in 
#         self.Function()
        
#         self.Function() takes initial state x0, input u0 and all model 
#         parameters as input. The model parameters can either be optimization
#         variables themselves (as in system identification) or the take specific 
#         values (when the estimated model is used for control)

#         Parameters
#         ----------
#         x0 : array-like with dimension [self.dim_x, 1]
#             initial state resp. state from last time-step
#         u0 : array-like with dimension [self.dim_u, 1]
#             input
#         params : dictionary, optional
#             params is None: This is the case during model based control,
#             self.Function() is evaluated with the numerical
#             values of the model parameters saved in self.Parameters
#             params is dictionary of opti.variables: During system identification
#             the model parameters are optimization variables themselves, so a 
#             dictionary of opti.variables is passed to self.Function()

#         Returns
#         -------
#         x1 : array-like with dimension [self.dim_x, 1]
#             output of the Feedforward Neural Network

#         """
#         if params==None:
#             params = self.Parameters
        
#         params_new = []
            
#         for name in  self.Function.name_in():
#             try:
#                 params_new.append(params[name])                      # Parameters are already in the right order as expected by Casadi Function
#             except:
#                 continue
        
#         x1 = self.Function(x0,u0,*params_new)     
                              
#         return x1
   
#     def Simulation(self,x0,u,params=None):
#         """
#         Repeated call of self.OneStepPrediction() for a given input trajectory
        

#         Parameters
#         ----------
#         x0 : array-like with dimension [self.dim_x, 1]
#             initial state resp
#         u : array-like with dimension [N,self.dim_u]
#             trajectory of input signal with length N
#         params : dictionary, optional
#             see self.OneStepPrediction()

#         Returns
#         -------
#         x : array-like with dimension [N+1,self.dim_x]
#             trajectory of output signal with length N+1 
            
#         """
        
#         x = []

#         # initial states
#         x.append(x0)
                      
#         # Simulate Model
#         for k in range(u.shape[0]):
#             x.append(self.OneStepPrediction(x[k],u[[k],:],params))
        
#         # Concatenate list to casadiMX
#         x = cs.hcat(x).T 
       
#         return x


    
# def logistic(x):
    
#     y = 0.5 + 0.5 * cs.tanh(0.5*x)

#     return y


    
    # def OneStepPrediction(self,c0,u0,params=None):
    #     """
    #     OneStepPrediction() evaluates the model equation defined in 
    #     self.Function()
        
    #     self.Function() takes initial cell-state c0, input u0 and all model 
    #     parameters as input. The model parameters can either be optimization
    #     variables themselves (as in system identification) or the take specific 
    #     values (when the estimated model is used for control)

    #     Parameters
    #     ----------
    #     c0 : array-like with dimension [self.dim_c, 1]
    #         initial cell-state resp. state from last time-step
    #     u0 : array-like with dimension [self.dim_u, 1]
    #         input
    #     params : dictionary, optional
    #         params is None: This is the case during model based control,
    #         self.Function() is evaluated with the numerical
    #         values of the model parameters saved in self.Parameters
    #         params is dictionary of opti.variables: During system identification
    #         the model parameters are optimization variables themselves, so a 
    #         dictionary of opti.variables is passed to self.Function()

    #     Returns
    #     -------
    #     c1 : array-like with dimension [self.dim_c, 1]
    #         new cell-state
    #     x1 : array-like with dimension [self.dim_x, 1]
    #         output of the Feedforward Neural Network
    #     """
    #     if params==None:
    #         params = self.Parameters
        
    #     params_new = []
            
    #     for name in  self.Function.name_in():
    #         try:
    #             params_new.append(params[name])                      # Parameters are already in the right order as expected by Casadi Function
    #         except:
    #             continue
        
    #     c1,x1 = self.Function(c0,u0,*params_new)     
                              
    #     return c1,x1
   
    # def Simulation(self,c0,u,params=None):
    #     """
    #     Repeated call of self.OneStepPrediction() for a given input trajectory
        

    #     Parameters
    #     ----------
    #     c0 : array-like with dimension [self.dim_c, 1]
    #         initial cell-state
    #     u : array-like with dimension [N,self.dim_u]
    #         trajectory of input signal with length N
    #     params : dictionary, optional
    #         see self.OneStepPrediction()

    #     Returns
    #     -------
    #     x : array-like with dimension [N+1,self.dim_x]
    #         trajectory of output signal with length N+1 
            
    #     """
        
    #     # Is that necessary?
    #     print('GRU Simulation ignores given initial state, initial state is set to zero!')
        
        
    #     c0 = np.zeros((self.dim_c,1))
        
    #     c = []
    #     x = []
        
    #     # initial cell state
    #     c.append(c0)
                      
    #     # Simulate Model
    #     for k in range(u.shape[0]):
    #         c_new,x_new = self.OneStepPrediction(c[k],u[k,:],params)
    #         c.append(c_new)
    #         x.append(x_new)
        
    #     # Concatenate list to casadiMX
    #     c = cs.hcat(c).T    
    #     x = cs.hcat(x).T
        
    #     return x[-1]