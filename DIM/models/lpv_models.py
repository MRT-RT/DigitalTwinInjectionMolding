# -*- coding: utf-8 -*-

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from DIM.optim.common import RK4
from .base import Recurrent
from .activations import *
from .layers import NN_layer, Eval_FeedForward_NN
from .initializations import XavierInitialization, RandomInitialization, HeInitialization

class LPV_RNN():
    '''
    Parent class for all Recurrent Neural Network LPV-Models
    '''

    def ParameterInitialization(self):
        '''
        Routine for parameter initialization. Takes input_names from the Casadi-
        Function defining the model equations self.Function and defines a 
        dictionary with input_names as keys. According to the initialization
        procedure defined in self.InitializationProcedure each key contains 
        a numpy array of appropriate shape. Also counts the number of model 
        parameters

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
        
        self.num_params = 0
        
        for p_name in self.Function.name_in()[2::]:
            
            p_shape = self.Function.size_in(p_name)
            self.Parameters[p_name] = initialization(p_shape)
            self.num_params = self.num_params + p_shape[0]*p_shape[1]
            
        # Initialize with specific inital parameters if given
        if self.InitialParameters is not None:
            for param in self.InitialParameters.keys():
                if param in self.Parameters.keys():
                    self.Parameters[param] = self.InitialParameters[param]

    def OneStepPrediction(self,x0,u0,params=None):
        '''
        Estimates the next state and output from current state and input
        x0: Casadi MX, current state
        u0: Casadi MX, current input
        params: A dictionary of opti variables, if the parameters of the model
                should be optimized, if None, then the current parameters of
                the model are used
        '''
        
        function_in = {'x':x0, 'u':u0}
        
        # If no parameters are provided the numerical values in self.Parameters
        # will be used
        if params==None:
            function_in.update(self.Parameters)
        
        
        # If parameters are provided, use them and use the numerical values 
        # in self.Parameters for the remaining not provided parameters   
        else:
                    
            for name in self.Function.name_in()[2::]:
                
                if name in params:
                    function_in[name] = params[name]
                else:
                    function_in[name] = self.Parameters[name]

        return self.Function(**function_in)   
   
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

        x = []
        y = []

        # initial states
        x.append(x0)
                      
        # Simulate Model
        for k in range(u.shape[0]):
            pred = self.OneStepPrediction(x[k],u[[k],:],params)
            x.append(pred['x_new'])
            y.append(pred['y_new'])
        
        # Concatenate list to casadiMX
        y = cs.hcat(y).T    
        x = cs.hcat(x).T
       
        return x,y    

    def EvalAffineParameters(self,x0,u0):
        '''

        '''
        
        params = self.Parameters
        
        params_new = []
            
        for name in self.AffineParameters.name_in():
            try:
                params_new.append(params[name])                      # Parameters are already in the right order as expected by Casadi Function
            except:
                continue
        
        theta = self.AffineParameters(x0,u0,*params_new)

        return theta

class RBFLPV(Recurrent):
    """
    Quasi-LPV model structure for system identification. Uses local linear models
    with nonlinear interpolation using RBFs. Scheduling variables are the
    input and the state.
    """
    def __init__(self,dim_u,dim_x,dim_y,u_label, y_label, name, dim_theta=0,
                 NN_dim=[],NN_act=[], initial_params=None, frozen_params=[], 
                 init_proc='random'):
        
        Recurrent.__init__(self, dim_u, dim_x, dim_y, u_label, y_label, name, 
                           initial_params, frozen_params, init_proc)
        
        self.NN_dim = NN_dim
        self.NN_act = NN_act
        self.dim_theta = dim_theta
               
        self.Initialize()

    def Initialize(self):
            
        # For convenience of notation
        dim_u = self.dim_u
        dim_x = self.dim_c 
        dim_y = self.dim_out   
        dim_theta = self.dim_theta
        NN_dim = self.NN_dim
        NN_act = self.NN_act

        # Define input, state and output vector
        u = cs.MX.sym('u',dim_u,1)
        x = cs.MX.sym('x',dim_x,1)
                    
        # Define Model Parameters
        A = cs.MX.sym('A',dim_x,dim_x,dim_theta)
        B = cs.MX.sym('B',dim_x,dim_u,dim_theta)
        C = cs.MX.sym('C',dim_y,dim_x,dim_theta)

        # Add parameters to list for cs function
        input = [x,u,*A,*B,*C]
        
        # Define the scheduling map
       
        NN = []
        
        # If u and the state itself are the scheduling signals
        if len(NN_dim)==0:
            c_u = cs.MX.sym('c_u',dim_u,1,dim_theta)
            c_x = cs.MX.sym('c_x',dim_x,1,dim_theta)
            w_u = cs.MX.sym('w_u',dim_u,1,dim_theta)
            w_x = cs.MX.sym('w_x',dim_x,1,dim_theta)                
            
            input.extend([*c_u,*c_x,*w_u,*w_x])
            
        # Else a NN is performing the scheduling map
        else:                
            
            for l in range(0,len(NN_dim)):
            
                if l == 0:
                    params = [cs.MX.sym('NN_Wx'+str(l),NN_dim[l],dim_x),
                              cs.MX.sym('NN_Wu'+str(l),NN_dim[l],dim_u),
                              cs.MX.sym('NN_b'+str(l),NN_dim[l],1)]
                else:
                    params = [cs.MX.sym('NN_W'+str(l),NN_dim[l],NN_dim[l-1]),
                              cs.MX.sym('NN_b'+str(l),NN_dim[l],1)]
                
                input.extend(params) 
                
                NN.append(params)
                
            c_h = cs.MX.sym('c_h',NN_dim[-1],1,dim_theta)
            w_h = cs.MX.sym('w_h',NN_dim[-1],1,dim_theta)
            
            input.extend([*c_h,*w_h]) 
                           
        # Define Model Equations, loop over all local models
        x_new = 0
        r_sum = 0
        
        for loc in range(0,len(A)):
            
            if len(NN)==0:
                c = cs.vertcat(c_x[loc],c_u[loc])
                w = cs.vertcat(w_x[loc],w_u[loc])
                
                r = RBF(cs.vertcat(x,u),c,w)
            else:
                # Calculate the activations of the NN
                NN_out = Eval_FeedForward_NN(cs.vertcat(x,u),NN,NN_act)
                
                r = RBF(NN_out[-1],c_h[loc],w_h[loc])
                
                
            x_new = x_new + \
            r * (cs.mtimes(A[loc],x) + cs.mtimes(B[loc],u)) # + O[loc])
            
            r_sum = r_sum + r
        
        x_new = x_new / (r_sum + 1e-20)
        
        y_new = 0
        r_sum = 0
        
        for loc in range(dim_theta):
            
            if len(NN)==0:
                c = cs.vertcat(c_x[loc],c_u[loc])
                w = cs.vertcat(w_x[loc],w_u[loc])
                
                r = RBF(cs.vertcat(x,u),c,w)
            else:
                # Calculate the activations of the NN
                NN_out = Eval_FeedForward_NN(cs.vertcat(x,u),NN,NN_act)
                
                r = RBF(NN_out[-1],c_h[loc],w_h[loc])  
                
            y_new = y_new + r * (cs.mtimes(C[loc],x_new))
            
            r_sum = r_sum + r
            
        y_new = y_new / (r_sum + 1e-20)
        
        # Define Casadi Function
       
        # Remove entries with dimension zero
        input = [i for i in input if all(i.shape)]
        input_names = [var.name() for var in input]
        
        output = [x_new,y_new]
        output_names = ['x_new','y_new']  
        
        self.Function = cs.Function(self.name, input, output, input_names,
                                    output_names)      
        
        self.ParameterInitialization()
        
        return None
        
       
    
    def AffineStateSpaceMatrices(self,theta):
        """
        A function that returns the state space matrices at a given value 
        for theta
        """
        
        return None #A,B,C

    def AffineParameters(self,x0,u0):
        '''

        '''
        
        params = self.Parameters
        
        params_new = []
            
        for name in self.AffineParameters.name_in():
            try:
                params_new.append(params[name])                      # Parameters are already in the right order as expected by Casadi Function
            except:
                continue
        
        theta = self.AffineParameters(x0,u0,*params_new) 

        return theta   

    def InitializeLocalModels(self,A,B,C,range_y=None,range_u=None):
        '''
        Initializes all local models with a given linear model and distributes
        the weighting functions uniformly over a given range
        A: array, system matrix
        B: array, input matrix
        C: array, output matrix
        op_range: array, expected operating range over which to distribute 
        the weighting functions
        '''

        self.Parameters['C'] = C
        
        for loc in range(0,self.dim_theta):
            
                i = str(loc)
                self.Parameters['A'+i] = A
                self.Parameters['B'+i] = B
                self.Parameters['C'+i] = C
                # initial_params['c_u'+i] = range_u[:,[0]] + \
                #     (range_u[:,[1]]-range_u[:,[0]]) * np.random.uniform(size=(self.dim_u,1))
                # initial_params['c_y'+i] = range_y[:,[0]] + \
                #     (range_y[:,[1]]-range_y[:,[0]]) * np.random.uniform(size=(self.dim_y,1))
        
        return None    


class RBFLPV_outputSched(Recurrent):
    """
    Quasi-LPV model structure for system identification. Uses local linear models
    with nonlinear interpolation using RBFs. Scheduling variables are the
    input and the output, which is assumed to be an observable linear combination
    of the states.
    """

    def __init__(self,dim_u,dim_x,dim_y,u_label, y_label, name, dim_theta=0,
                 NN_dim=[],NN_act=[], initial_params=None, frozen_params=[], 
                 init_proc='random'):
        
        Recurrent.__init__(self, dim_u, dim_x, dim_y, u_label, y_label, name, 
                           initial_params, frozen_params, init_proc)
        
        self.NN_dim = NN_dim
        self.NN_act = NN_act
        self.dim_theta = dim_theta
               
        self.Initialize()

    def Initialize(self):
            
        # For convenience of notation
        dim_u = self.dim_u
        dim_x = self.dim_c 
        dim_y = self.dim_out   
        dim_theta = self.dim_theta
        
        NN_dim = self.NN_dim
        NN_act = self.NN_act
        
        name = self.name
        
        # Define input, state and output vector
        u = cs.MX.sym('u',dim_u,1)
        x = cs.MX.sym('x',dim_x,1)
                    
        # Define Model Parameters
        A = cs.MX.sym('A',dim_x,dim_x,dim_theta)
        B = cs.MX.sym('B',dim_x,dim_u,dim_theta)
        C = cs.MX.sym('C',dim_y,dim_x)

        # Add parameters to list for cs function
        input = [x,u,*A,*B,C]
        
        # Define the scheduling map
        NN = []
        
        # If u and the state itself are the scheduling signals
        if len(NN_dim)==0:
            c_u = cs.MX.sym('c_u',dim_u,1,dim_theta)
            c_y = cs.MX.sym('c_y',dim_y,1,dim_theta)
            w_u = cs.MX.sym('w_u',dim_u,1,dim_theta)
            w_y = cs.MX.sym('w_y',dim_y,1,dim_theta)                
            
            input.extend([*c_u,*c_y,*w_u,*w_y])
            
        # Else a NN is performing the scheduling map
        else:                
            
            for l in range(0,len(NN_dim)):
            
                if l == 0:
                    params = [cs.MX.sym('NN_Wy'+str(l),NN_dim[l],dim_y),
                              cs.MX.sym('NN_Wu'+str(l),NN_dim[l],dim_u),
                              cs.MX.sym('NN_b'+str(l),NN_dim[l],1)]
                else:
                    params = [cs.MX.sym('NN_W'+str(l),NN_dim[l],NN_dim[l-1]),
                              cs.MX.sym('NN_b'+str(l),NN_dim[l],1)]
                    
                input.extend(params) 
                
                NN.append(params)
                
            c_h = cs.MX.sym('c_h',NN_dim[-1],1,dim_theta)
            w_h = cs.MX.sym('w_h',NN_dim[-1],1,dim_theta)
             
            input.extend([*c_h,*w_h]) 
            
        # Define Model Equations, loop over all local models
        y = cs.mtimes(C,x)
        
        x_new = 0
        r_sum = 0
        
        for loc in range(0,len(A)):
            
            if len(NN)==0:
                c = cs.vertcat(c_y[loc],c_u[loc])
                w = cs.vertcat(w_y[loc],w_u[loc])
                
                r = RBF(cs.vertcat(y,u),c,w)
            else:
                # Calculate the activations of the NN
                NN_out = Eval_FeedForward_NN(cs.vertcat(y,u),NN,NN_act)
                
                r = RBF(NN_out[-1],c_h[loc],w_h[loc])
                
                
            x_new = x_new + \
            r * (cs.mtimes(A[loc],x) + cs.mtimes(B[loc],u)) # + O[loc])
            
            r_sum = r_sum + r
        
        x_new = x_new / (r_sum + 1e-20)
        
        y_new = cs.mtimes(C,x_new)
        
        # Define Casadi Function
       
        # Define input of Casadi Function and save all parameters in 
        # dictionary
        
        # Remove entries with dimension zero
        input = [i for i in input if all(i.shape)]
        input_names = [var.name() for var in input]
             
          

        output = [x_new,y_new]
        output_names = ['x_new','y_new']  
        
        self.Function = cs.Function(name, input, output, input_names,output_names)
            
        self.ParameterInitialization()            
            
        return None        

   
    def InitializeLocalModels(self,A,B,C,range_y=None,range_u=None):
        '''
        Initializes all local models with a given linear model and distributes
        the weighting functions uniformly over a given range
        A: array, system matrix
        B: array, input matrix
        C: array, output matrix
        op_range: array, expected operating range over which to distribute 
        the weighting functions
        '''

        self.Parameters['C'] = C
        
        for loc in range(0,self.dim_theta):
            
                i = str(loc)
                self.Parameters['A'+i] = A
                self.Parameters['B'+i] = B
                
                # initial_params['C'+i] = C
                # initial_params['c_u'+i] = range_u[:,[0]] + \
                #     (range_u[:,[1]]-range_u[:,[0]]) * np.random.uniform(size=(self.dim_u,1))
                # initial_params['c_y'+i] = range_y[:,[0]] + \
                #     (range_y[:,[1]]-range_y[:,[0]]) * np.random.uniform(size=(self.dim_y,1))
        
        return None
        
    
    def AffineStateSpaceMatrices(self,theta):
        """
        A function that returns the state space matrices at a given value 
        for theta
        """
        # A_0 = self.Parameters['A_0']
        # B_0 = self.Parameters['B_0']
        # C_0 = self.Parameters['C_0']
    
        # A_lpv = self.Parameters['A_0']
        # B_lpv = self.Parameters['B_lpv']
        # C_lpv = self.Parameters['C_lpv']  
    
        # W_A = self.Parameters['W_A']
        # W_B = self.Parameters['W_B']
        # W_C = self.Parameters['W_C']      
    
        # theta_A = theta[0:self.dim_thetaA]
        # theta_B = theta[self.dim_thetaA:self.dim_thetaA+self.dim_thetaB]
        # theta_C = theta[self.dim_thetaA+self.dim_thetaB:self.dim_thetaA+
        #                 self.dim_thetaB+self.dim_thetaC]
        
        # A = A_0 + np.linalg.multi_dot([A_lpv,np.diag(theta_A),W_A])
        # B = B_0 + np.linalg.multi_dot([B_lpv,np.diag(theta_B),W_B])
        # C = C_0 + np.linalg.multi_dot([C_lpv,np.diag(theta_C),W_C]) 
        
        return None #A,B,C

    
class RehmerLPV(Recurrent):
    """
    Quasi-LPV model structure for system identification. Uses a structured
    RNN with "deep" gates that can be transformed into an affine quasi LPV
    representation. Scheduling variables are the input and the state.
    """

    def __init__(self,dim_u,dim_x,dim_y,u_label,y_label,name,dim_thetaA=0,
                 dim_thetaB=0,dim_thetaC=0, NN_1_dim=[],NN_2_dim=[],
                 NN_3_dim=[],NN1_act=[],NN2_act=[],
                 NN3_act=[], frozen_params = [],initial_params=None,
                 init_proc='random'):
        '''
        Initializes the model structure by Rehmer et al. 2021.
        dim_u: int, dimension of the input vector
        dim_x: int, dimension of the state vector
        dim_y: int, dimension of the output vector
        dim_thetaA: int, dimension of the affine parameter associated with the 
        system matrix
        dim_thetaB: int, dimension of the affine parameter associated with the 
        input matrix
        dim_thetaC: int, dimension of the affine parameter associated with the 
        output matrix
        NN_1_dim: list, each entry is an integer specifying the number of neurons 
        in the hidden layers of the NN associated with the system matrix
        NN_2_dim: list, each entry is an integer specifying the number of neurons 
        in the hidden layers of the NN associated with the input matrix      
        NN_3_dim: list, each entry is an integer specifying the number of neurons 
        in the hidden layers of the NN associated with the system matrix     
        
        activation: list, each entry is an integer, that specifies the
        activation function used in the layers of the NNs
                    0 --> tanh()
                    1 --> logistic()
                    2 --> linear()
        initial_params: dict, dictionary specifying the inital parameter values
        name: str, specifies name of the model
        '''
        
        Recurrent.__init__(self, dim_u, dim_x, dim_y, u_label, y_label, name, 
                           initial_params, frozen_params, init_proc)
 
        self.dim_thetaA = dim_thetaA
        self.dim_thetaB = dim_thetaB
        self.dim_thetaC = dim_thetaC
        
        self.dim = dim_thetaA+dim_thetaB+dim_thetaC
        
        self.NN_1_dim = NN_1_dim
        self.NN_2_dim = NN_2_dim
        self.NN_3_dim = NN_3_dim
        
        self.NN1_act = NN1_act
        self.NN2_act = NN2_act
        self.NN3_act = NN3_act
        

        self.Initialize()

    def Initialize(self):
            
        # For convenience of notation
        dim_u = self.dim_u
        dim_x = self.dim_c 
        dim_y = self.dim_out   
        dim_thetaA = self.dim_thetaA
        dim_thetaB = self.dim_thetaB
        dim_thetaC = self.dim_thetaC
        NN_1_dim = self.NN_1_dim
        NN_2_dim = self.NN_2_dim
        NN_3_dim = self.NN_3_dim    
        NN1_act = self.NN1_act
        NN2_act = self.NN2_act
        NN3_act = self.NN3_act
       
        name = self.name
        
        # Define input, state and output vector
        u = cs.MX.sym('u',dim_u,1)
        x = cs.MX.sym('x',dim_x,1)
        
        # Define Model Parameters for the linear part
        A_0 = cs.MX.sym('A_0',dim_x,dim_x)
        B_0 = cs.MX.sym('B_0',dim_x,dim_u)
        C_0 = cs.MX.sym('C_0',dim_y,dim_x)
        
        # Define Model Parameters for the time varying part by Lachhab
        A_1 = cs.MX.sym('A_1',dim_x,dim_thetaA)
        E_1 = cs.MX.sym('E_1',dim_thetaA,dim_x)
  
        B_1 = cs.MX.sym('B_1',dim_x,dim_thetaB)
        E_2 = cs.MX.sym('E_2',dim_thetaB,dim_u)

        C_1 = cs.MX.sym('C_1',dim_y,dim_thetaC)
        E_3 = cs.MX.sym('E_3',dim_thetaC,dim_x)            
        
        # Define Parameters for the multiplicative Neural Networks by Rehmer
        NN1 = []
        NN2 = []
        NN3 = []
        
        # NN_1_dim.append(dim_thetaA)
        for NN, NN_name, NN_dim in zip([NN1,NN2,NN3],['NN1','NN2','NN3'],
                                       [NN_1_dim,NN_2_dim,NN_3_dim]):
            
            for l in range(0,len(NN_dim)):
            
                if l == 0:
                    params = [cs.MX.sym(NN_name+'_Wx'+str(l),NN_dim[l],dim_x),
                              cs.MX.sym(NN_name+'_Wu'+str(l),NN_dim[l],dim_u),
                              cs.MX.sym(NN_name+'_b'+str(l),NN_dim[l],1)]
                else:
                    params = [cs.MX.sym(NN_name+'_W'+str(l),NN_dim[l],NN_dim[l-1]),
                              cs.MX.sym(NN_name+'_b'+str(l),NN_dim[l],1)]
                
                NN.append(params)
        
       
        # Define Model Equations
       
        # Calculate the activations of the NNs by looping over each NN and
        # each layer
        NN_out = []
        
        for NN,NN_act in zip([NN1,NN2,NN3],[NN1_act,NN2_act,NN3_act]):
            
            out = Eval_FeedForward_NN(cs.vertcat(x,u),NN,NN_act)
            
            if out:
                NN_out.append(out)
            else:
                NN_out.append([0])
            
        # State and output equation
        x_new = cs.mtimes(A_0,x) + cs.mtimes(B_0,u) + cs.mtimes(A_1, 
                NN_out[0][-1]*cs.tanh(cs.mtimes(E_1,x))) + cs.mtimes(B_1, 
                NN_out[1][-1]*cs.tanh(cs.mtimes(E_2,u)))
        y_new = cs.mtimes(C_0,x_new) + cs.mtimes(C_1, 
                NN_out[2][-1]*cs.tanh(cs.mtimes(E_3,x_new)))
        
        
        # Define inputs and outputs for casadi function
        input = [x,u,A_0,A_1,E_1,B_0,B_1,E_2,C_0,C_1,E_3]
        input_names = ['x','u','A_0','A_1','E_1','B_0','B_1','E_2','C_0',
                       'C_1','E_3']
        
       
        # Add remaining parameters in loop since they depend on depth of NNs
        for NN_name, NN in zip(['NN1','NN2','NN3'],[NN1,NN2,NN3]):
            for l in range(0,len(NN)):
                input.extend(NN[l])

        # Remove entries with dimension zero
        input = [i for i in input if all(i.shape)]
        input_names = [var.name() for var in input]        
       
        output = [x_new,y_new]
        output_names = ['x_new','y_new']
        
        self.Function = cs.Function(name, input, output, input_names,
                                    output_names)
       
        # Calculate affine parameters
        theta_A = NN_out[0][-1] * cs.tanh(cs.mtimes(E_1,x))/cs.mtimes(E_1,x)
        theta_B = NN_out[1][-1] * cs.tanh(cs.mtimes(E_2,u))/cs.mtimes(E_2,u)
        theta_C = NN_out[2][-1] * cs.tanh(cs.mtimes(E_3,x))/cs.mtimes(E_3,x)
        
        theta = cs.vertcat(theta_A,theta_B,theta_C)   
        
        self.AffineParameters = cs.Function('AffineParameters',input,
                                            [theta],input_names,['theta'])
        
        # Initialize symbolic variables with numeric values
        self.ParameterInitialization()
        
        return None
    

    
        
    def AffineStateSpaceMatrices(self,theta):
        
        A_0 = self.Parameters['A_0']
        B_0 = self.Parameters['B_0']
        C_0 = self.Parameters['C_0']
    
        A_lpv = self.Parameters['A_0']
        B_lpv = self.Parameters['B_lpv']
        C_lpv = self.Parameters['C_lpv']  
    
        W_A = self.Parameters['W_A']
        W_B = self.Parameters['W_B']
        W_C = self.Parameters['W_C']      
    
        theta_A = theta[0:self.dim_thetaA]
        theta_B = theta[self.dim_thetaA:self.dim_thetaA+self.dim_thetaB]
        theta_C = theta[self.dim_thetaA+self.dim_thetaB:self.dim_thetaA+
                        self.dim_thetaB+self.dim_thetaC]
        
        A = A_0 + np.linalg.multi_dot([A_lpv,np.diag(theta_A),W_A])
        B = B_0 + np.linalg.multi_dot([B_lpv,np.diag(theta_B),W_B])
        C = C_0 + np.linalg.multi_dot([C_lpv,np.diag(theta_C),W_C]) 
        
        return A,B,C

class RehmerLPV_outputSched(Recurrent):
    """
    Quasi-LPV model structure for system identification. Uses a structured
    RNN with "deep" gates that can be transformed into an affine quasi LPV
    representation. Scheduling variables are the input and the output,
    which is assumed to be a linear and observable combination of the states.
    """

    def __init__(self,dim_u,dim_x,dim_y,u_label,y_label,name,dim_thetaA=0,
                 dim_thetaB=0,dim_thetaC=0, NN_1_dim=[],NN_2_dim=[],
                 NN_3_dim=[],NN1_act=[],NN2_act=[],
                 NN3_act=[], frozen_params = [],initial_params=None,
                 init_proc='random'):
        '''
        Initializes the model structure by Rehmer et al. 2021.
        dim_u: int, dimension of the input vector
        dim_x: int, dimension of the state vector
        dim_y: int, dimension of the output vector
        dim_thetaA: int, dimension of the affine parameter associated with the 
        system matrix
        dim_thetaB: int, dimension of the affine parameter associated with the 
        input matrix
        dim_thetaC: int, dimension of the affine parameter associated with the 
        output matrix
        NN_1_dim: list, each entry is an integer specifying the number of neurons 
        in the hidden layers of the NN associated with the system matrix
        NN_2_dim: list, each entry is an integer specifying the number of neurons 
        in the hidden layers of the NN associated with the input matrix      
        NN_3_dim: list, each entry is an integer specifying the number of neurons 
        in the hidden layers of the NN associated with the system matrix     
        
        activation: list, each entry is an integer, that specifies the
        activation function used in the layers of the NNs
                    0 --> tanh()
                    1 --> logistic()
                    2 --> linear()
        initial_params: dict, dictionary specifying the inital parameter values
        name: str, specifies name of the model
        '''
        
        Recurrent.__init__(self, dim_u, dim_x, dim_y, u_label, y_label, name, 
                           initial_params, frozen_params, init_proc)
 
        self.dim_thetaA = dim_thetaA
        self.dim_thetaB = dim_thetaB
        self.dim_thetaC = dim_thetaC
        
        self.dim = dim_thetaA+dim_thetaB+dim_thetaC
        
        self.NN_1_dim = NN_1_dim
        self.NN_2_dim = NN_2_dim
        self.NN_3_dim = NN_3_dim
        
        self.NN1_act = NN1_act
        self.NN2_act = NN2_act
        self.NN3_act = NN3_act
        

        self.Initialize()

    def Initialize(self):
            
        # For convenience of notation
        dim_u = self.dim_u
        dim_x = self.dim_c 
        dim_y = self.dim_out   
        dim_thetaA = self.dim_thetaA
        dim_thetaB = self.dim_thetaB
        dim_thetaC = self.dim_thetaC
        
        NN_1_dim = self.NN_1_dim
        NN_2_dim = self.NN_2_dim
        NN_3_dim = self.NN_3_dim
        
        NN1_act = self.NN1_act
        NN2_act = self.NN2_act
        NN3_act = self.NN3_act
        
        name = self.name
        
        # Define input, state and output vector
        u = cs.MX.sym('u',dim_u,1)
        x = cs.MX.sym('x',dim_x,1)
        
        # Define Model Parameters for the linear part
        A_0 = cs.MX.sym('A_0',dim_x,dim_x)
        B_0 = cs.MX.sym('B_0',dim_x,dim_u)
        C_0 = cs.MX.sym('C_0',dim_y,dim_x)
        
        # Define Model Parameters for the time varying part by Lachhab
        A_1 = cs.MX.sym('A_1',dim_x,dim_thetaA)
        E_1 = cs.MX.sym('E_1',dim_thetaA,dim_y)
  
        B_1 = cs.MX.sym('B_1',dim_x,dim_thetaB)
        E_2 = cs.MX.sym('E_2',dim_thetaB,dim_u)
       
        # Define Parameters for the multiplicative Neural Networks by Rehmer
        NN1 = []
        NN2 = []
        
        # NN_1_dim.append(dim_thetaA)
        for NN, NN_name, NN_dim in zip([NN1,NN2],['NN1','NN2'],
                                       [NN_1_dim,NN_2_dim]):
            
            for l in range(0,len(NN_dim)):
            
                if l == 0:
                    params = [cs.MX.sym(NN_name+'_Wy'+str(l),NN_dim[l],dim_y),
                              cs.MX.sym(NN_name+'_Wu'+str(l),NN_dim[l],dim_u),
                              cs.MX.sym(NN_name+'_b'+str(l),NN_dim[l],1)]
                else:
                    params = [cs.MX.sym(NN_name+'_W'+str(l),NN_dim[l],NN_dim[l-1]),
                              cs.MX.sym(NN_name+'_b'+str(l),NN_dim[l],1)]
                
                NN.append(params)
                

        # Define Model Equations
        
        # Output from last state
        y = cs.mtimes(C_0,x)
       
        # Calculate the activations of the NNs by looping over each NN and
        # each layer
        # NN_out = [[0],[0]]
        NN_out = []
        
        for NN,NN_act in zip([NN1,NN2],[NN1_act,NN2_act]):

            out = Eval_FeedForward_NN(cs.vertcat(y,u),NN,NN_act)
            
            if out:
                NN_out.append(out)
            else:
                NN_out.append([0])

        # State and output equation
        x_new = cs.mtimes(A_0,x) + cs.mtimes(B_0,u) + cs.mtimes(A_1, 
                NN_out[0][-1]*cs.tanh(cs.mtimes(E_1,y))) + cs.mtimes(B_1, 
                NN_out[1][-1]*cs.tanh(cs.mtimes(E_2,u)))
        y_new = cs.mtimes(C_0,x_new)
        
        
        # Define inputs and outputs for casadi function
        input = [x,u,A_0,A_1,E_1,B_0,B_1,E_2,C_0]
        
        # Add remaining parameters in loop since they depend on depth of NNs
        for NN_name, NN in zip(['NN1','NN2'],[NN1,NN2]):
            for l in range(0,len(NN)):
                input.extend(NN[l])
        
            
        # Remove entries with dimension zero
        input = [i for i in input if all(i.shape)]
        input_names = [var.name() for var in input]
        
        output = [x_new,y_new]
        output_names = ['x_new','y_new']
        
        self.Function = cs.Function(name, input, output, input_names,
                                    output_names)
               
        # Calculate affine parameters
        theta_A = NN_out[0][-1] * cs.tanh(cs.mtimes(E_1,y))/cs.mtimes(E_1,y)
        theta_B = NN_out[1][-1] * cs.tanh(cs.mtimes(E_2,u))/cs.mtimes(E_2,u)
        
        theta = cs.vertcat(theta_A,theta_B)   
        
        self.AffineParameters = cs.Function('AffineParameters',input,
                                            [theta],input_names,['theta'])
        
         # Initialize symbolic variables with numeric values
        self.ParameterInitialization()
        
        return None
       
    def AffineStateSpaceMatrices(self,theta):
        
        A_0 = self.Parameters['A_0']
        B_0 = self.Parameters['B_0']
        C_0 = self.Parameters['C_0']
    
        A_lpv = self.Parameters['A_lpv']
        B_lpv = self.Parameters['B_lpv']
        C_lpv = self.Parameters['C_lpv']  
    
        W_A = self.Parameters['W_A']
        W_B = self.Parameters['W_B']
        W_C = self.Parameters['W_C']      
    
        theta_A = theta[0:self.dim_thetaA]
        theta_B = theta[self.dim_thetaA:self.dim_thetaA+self.dim_thetaB]
        theta_C = theta[self.dim_thetaA+self.dim_thetaB:self.dim_thetaA+
                        self.dim_thetaB+self.dim_thetaC]
        
        A = A_0 + np.linalg.multi_dot([A_lpv,np.diag(theta_A),W_A])
        B = B_0 + np.linalg.multi_dot([B_lpv,np.diag(theta_B),W_B])
        C = C_0 + np.linalg.multi_dot([C_lpv,np.diag(theta_C),W_C]) 
        
        return A,B,C


class RehmerLPV_old(Recurrent):

    def __init__(self,dim_u,dim_x,dim_y,u_label, y_label, name,
                 dim_thetaA=0,dim_thetaB=0,dim_thetaC=0,fA_dim=0,fB_dim=0,
                 fC_dim=0,activation=0,initial_params=None,frozen_params = [],
                 init_proc='random'):
        

        '''
        Initializes the model structure by Rehmer et al. 2021.
        dim_u: int, dimension of the input vector
        dim_x: int, dimension of the state vector
        dim_y: int, dimension of the output vector
        dim_thetaA: int, dimension of the affine parameter associated with the 
        system matrix
        dim_thetaB: int, dimension of the affine parameter associated with the 
        input matrix
        dim_thetaC: int, dimension of the affine parameter associated with the 
        output matrix
        fA_dim: int, number of neurons in the hidden layer of the NN associated 
        with the system matrix
        fB_dim: int, number of neurons in the hidden layer of the NN associated 
        with the input matrix        
        fC_dim: int, number of neurons in the hidden layer of the NN associated 
        with the output matrix        
        
        activation: int, specifies activation function used in the NNs
                    0 --> tanh()
                    1 --> logistic()
                    2 --> linear()
        initial_params: dict, dictionary specifying the inital parameter values
        name: str, specifies name of the model
        '''

        Recurrent.__init__(self, dim_u, dim_x, dim_y, u_label, y_label, name, 
                           initial_params, frozen_params, init_proc)

        self.dim_thetaA = dim_thetaA
        self.dim_thetaB = dim_thetaB
        self.dim_thetaC = dim_thetaC
        self.fA_dim = fA_dim
        self.fB_dim = fB_dim
        self.fC_dim = fC_dim
        self.activation = activation

        self.dim = dim_thetaA+dim_thetaB+dim_thetaC
        
        self.Initialize()


    def Initialize(self,initial_params=None):
            
            # For convenience of notation
            dim_u = self.dim_u
            dim_x = self.dim_c 
            dim_y = self.dim_out   
            dim_thetaA = self.dim_thetaA
            dim_thetaB = self.dim_thetaB
            dim_thetaC = self.dim_thetaC
            fA_dim = self.fA_dim
            fB_dim = self.fB_dim
            fC_dim = self.fC_dim    
           
            name = self.name
            
            # Define input, state and output vector
            u = cs.MX.sym('u',dim_u,1)
            x = cs.MX.sym('x',dim_x,1)
            y = cs.MX.sym('y',dim_y,1)
            
            # Define Model Parameters
            A_0 = cs.MX.sym('A_0',dim_x,dim_x)
            A_lpv = cs.MX.sym('A_lpv',dim_x,dim_thetaA)
            W_A = cs.MX.sym('W_A',dim_thetaA,dim_x)
            
            W_fA_x = cs.MX.sym('W_fA_x',fA_dim,dim_x)
            W_fA_u = cs.MX.sym('W_fA_u',fA_dim,dim_u)
            b_fA_h = cs.MX.sym('b_fA_h',fA_dim,1)
            W_fA = cs.MX.sym('W_fA',dim_thetaA,fA_dim)
            b_fA = cs.MX.sym('b_fA',dim_thetaA,1)
            
            B_0 = cs.MX.sym('B_0',dim_x,dim_u)
            B_lpv = cs.MX.sym('B_lpv',dim_x,dim_thetaB)
            W_B = cs.MX.sym('W_B',dim_thetaB,dim_u)
  
            W_fB_x = cs.MX.sym('W_fB_x',fB_dim,dim_x)
            W_fB_u = cs.MX.sym('W_fB_u',fB_dim,dim_u)
            b_fB_h = cs.MX.sym('b_fB_h',fB_dim,1)
            W_fB = cs.MX.sym('W_fB',dim_thetaB,fB_dim)
            b_fB = cs.MX.sym('b_fB',dim_thetaB,1)            
  
            C_0 = cs.MX.sym('C_0',dim_y,dim_x)
            C_lpv = cs.MX.sym('C_lpv',dim_y,dim_thetaC)
            W_C = cs.MX.sym('W_C',dim_thetaC,dim_x)
            
            W_fC_x = cs.MX.sym('W_fC_x',fC_dim,dim_x)
            W_fC_u = cs.MX.sym('W_fC_u',fC_dim,dim_u)
            b_fC_h = cs.MX.sym('b_fC_h',fC_dim,1)
            W_fC = cs.MX.sym('W_fC',dim_thetaC,fC_dim)
            b_fC = cs.MX.sym('b_fC',dim_thetaC,1)            
            
            # Define Model Equations
            fA_h = cs.tanh(cs.mtimes(W_fA_x,x) + cs.mtimes(W_fA_u,u) + b_fA_h)
            fA = logistic(cs.mtimes(W_fA,fA_h)+b_fA)
            
            fB_h = cs.tanh(cs.mtimes(W_fB_x,x) + cs.mtimes(W_fB_u,u) + b_fB_h)
            fB = logistic(cs.mtimes(W_fB,fB_h)+b_fB)
            
            fC_h = cs.tanh(cs.mtimes(W_fC_x,x) + cs.mtimes(W_fC_u,u) + b_fC_h)
            fC = logistic(cs.mtimes(W_fC,fC_h)+b_fC)
            
            x_new = cs.mtimes(A_0,x) + cs.mtimes(B_0,u) + cs.mtimes(A_lpv, 
                    fA*cs.tanh(cs.mtimes(W_A,x))) + cs.mtimes(B_lpv, 
                    fB*cs.tanh(cs.mtimes(W_B,u)))
            y_new = cs.mtimes(C_0,x_new) + cs.mtimes(C_lpv, 
                    fC*cs.tanh(cs.mtimes(W_C,x_new)))
            
            input = [x,u,A_0,A_lpv,W_A,W_fA_x,W_fA_u,b_fA_h,W_fA,b_fA,
                      B_0,B_lpv,W_B,W_fB_x,W_fB_u,b_fB_h,W_fB,b_fB,
                      C_0,C_lpv,W_C,W_fC_x,W_fC_u,b_fC_h,W_fC,b_fC]
            
            # Remove entries with dimension zero
            input = [i for i in input if all(i.shape)]
            input_names = [var.name() for var in input]
            
            output = [x_new,y_new]
            output_names = ['x_new','y_new']
            
            self.Function = cs.Function(name, input, output, input_names,
                                        output_names)
            
            
            # Calculate affine parameters
            theta_A = fA * cs.tanh(cs.mtimes(W_A,x))/(cs.mtimes(W_A,x)+1e-6)
            theta_B = fB * cs.tanh(cs.mtimes(W_B,u))/(cs.mtimes(W_B,u)+1e-6)
            theta_C = fC * cs.tanh(cs.mtimes(W_C,x))/(cs.mtimes(W_C,x)+1e-6)
            
            theta = cs.vertcat(theta_A,theta_B,theta_C)   
            
            self.AffineParameters = cs.Function('AffineParameters',input,
                                                [theta],input_names,['theta'])
            
            
            self.ParameterInitialization()
            
            return None
        
    def AffineStateSpaceMatrices(self,theta):
        
        A_0 = self.Parameters['A_0']
        B_0 = self.Parameters['B_0']
        C_0 = self.Parameters['C_0']
    
        A_lpv = self.Parameters['A_lpv']
        B_lpv = self.Parameters['B_lpv']
        C_lpv = self.Parameters['C_lpv']  
    
        W_A = self.Parameters['W_A']
        W_B = self.Parameters['W_B']
        W_C = self.Parameters['W_C']      
    
        theta_A = theta[0:self.dim_thetaA]
        theta_B = theta[self.dim_thetaA:self.dim_thetaA+self.dim_thetaB]
        theta_C = theta[self.dim_thetaA+self.dim_thetaB:self.dim_thetaA+
                        self.dim_thetaB+self.dim_thetaC]
        
        A = A_0 + np.linalg.multi_dot([A_lpv,np.diag(theta_A),W_A])
        B = B_0 + np.linalg.multi_dot([B_lpv,np.diag(theta_B),W_B])
        C = C_0 + np.linalg.multi_dot([C_lpv,np.diag(theta_C),W_C]) 
        
        return A,B,C

    def EvalAffineParameters(self,x0,u0,params=None):
        '''

        '''
        if params==None:
            params = self.Parameters        
        
        params_new = []
            
        for name in self.AffineParameters.name_in():
            try:
                params_new.append(params[name])                      # Parameters are already in the right order as expected by Casadi Function
            except:
                continue
        
        theta = self.AffineParameters(x0,u0,*params_new)

        return theta


class LachhabLPV(Recurrent):
    """
    Quasi-LPV model structure for system identification. Uses a structured
    RNN which can be transformed into an affine LPV representation. Scheduling variables are the
    input and the state.
    """

    def __init__(self,dim_u,dim_x,dim_y,u_label, y_label, name,
                 dim_thetaA=0,dim_thetaB=0,dim_thetaC=0,
                 initial_params=None, frozen_params = [], init_proc='random'):
        
        
        Recurrent.__init__(self, dim_u, dim_x, dim_y, u_label, y_label, name, 
                           initial_params, frozen_params, init_proc)
             
        self.dim_thetaA = dim_thetaA
        self.dim_thetaB = dim_thetaB
        self.dim_thetaC = dim_thetaC
        
        self.dim = dim_thetaA+dim_thetaB+dim_thetaC
        
        self.Initialize()

    def Initialize(self):
            
            # For convenience of notation
            dim_u = self.dim_u
            dim_x = self.dim_c 
            dim_y = self.dim_out   
            dim_thetaA = self.dim_thetaA
            dim_thetaB = self.dim_thetaB
            dim_thetaC = self.dim_thetaC
            name = self.name
            
            # Define input, state and output vector
            u = cs.MX.sym('u',dim_u,1)
            x = cs.MX.sym('x',dim_x,1)
            
            # Define Model Parameters
            A_0 = cs.MX.sym('A_0',dim_x,dim_x)
            A_lpv = cs.MX.sym('A_lpv',dim_x,dim_thetaA)
            W_A = cs.MX.sym('W_A',dim_thetaA,dim_x)
            
            B_0 = cs.MX.sym('B_0',dim_x,dim_u)
            B_lpv = cs.MX.sym('B_lpv',dim_x,dim_thetaB)
            W_B = cs.MX.sym('W_B',dim_thetaB,dim_u)
            
            C_0 = cs.MX.sym('C_0',dim_y,dim_x)
            C_lpv = cs.MX.sym('C_lpv',dim_y,dim_thetaC)
            W_C = cs.MX.sym('W_C',dim_thetaC,dim_x)
           
            # Define Model Equations
            x_new = cs.mtimes(A_0,x) + cs.mtimes(B_0,u) + cs.mtimes(A_lpv, 
                    cs.tanh(cs.mtimes(W_A,x))) + cs.mtimes(B_lpv, 
                    cs.tanh(cs.mtimes(W_B,u)))
            y_new = cs.mtimes(C_0,x_new) + cs.mtimes(C_lpv, 
                    cs.tanh(cs.mtimes(W_C,x_new)))
            
            
            input = [x,u,A_0,A_lpv,W_A,B_0,B_lpv,W_B,C_0,C_lpv,W_C]
            
            # Remove entries with dimension zero
            input = [i for i in input if all(i.shape)]
            input_names = [var.name() for var in input]
            
            output = [x_new,y_new]
            output_names = ['x_new','y_new']  
            
            self.Function = cs.Function(name, input, output, input_names,output_names)

            # Calculate affine parameters
            theta_A = cs.tanh(cs.mtimes(W_A,x)/(cs.mtimes(W_A,x)))
            theta_B = cs.tanh(cs.mtimes(W_B,u)/(cs.mtimes(W_B,u)))
            theta_C = cs.tanh(cs.mtimes(W_C,x)/(cs.mtimes(W_C,x)))
            
            theta = cs.vertcat(theta_A,theta_B,theta_C)   
            
            self.AffineParameters = cs.Function('AffineParameters',input,
                                                [theta],input_names,['theta'])

            self.ParameterInitialization()
            
            return None

class LachhabLPV_outputSched(Recurrent):
    """
    Quasi-LPV model structure for system identification. Uses a structured
    RNN which can be transformed into an affine LPV representation. Scheduling 
    variables are the input and the output, which is assumed to be a linear 
    observable combination of the state.
    """
       
    def __init__(self,dim_u,dim_x,dim_y,u_label, y_label, name,
                 dim_thetaA=0,dim_thetaB=0,dim_thetaC=0,
                 initial_params=None, frozen_params = [], init_proc='random'):
        
        
        Recurrent.__init__(self, dim_u, dim_x, dim_y, u_label, y_label, name, 
                           initial_params, frozen_params, init_proc)
             
        self.dim_thetaA = dim_thetaA
        self.dim_thetaB = dim_thetaB
        self.dim_thetaC = dim_thetaC
        
        self.dim = dim_thetaA+dim_thetaB+dim_thetaC
        
        self.Initialize()

    def Initialize(self):
            
            # For convenience of notation
            dim_u = self.dim_u
            dim_x = self.dim_c
            dim_y = self.dim_out   
            dim_thetaA = self.dim_thetaA
            dim_thetaB = self.dim_thetaB

            name = self.name
            
            # Define input, state and output vector
            u = cs.MX.sym('u',dim_u,1)
            x = cs.MX.sym('x',dim_x,1)
            
            # Define Model Parameters
            A_0 = cs.MX.sym('A_0',dim_x,dim_x)
            A_lpv = cs.MX.sym('A_lpv',dim_x,dim_thetaA)
            W_A = cs.MX.sym('W_A',dim_thetaA,dim_y)
            
            B_0 = cs.MX.sym('B_0',dim_x,dim_u)
            B_lpv = cs.MX.sym('B_lpv',dim_x,dim_thetaB)
            W_B = cs.MX.sym('W_B',dim_thetaB,dim_u)
            
            C_0 = cs.MX.sym('C_0',dim_y,dim_x)

            # Define Model Equations
            y = cs.mtimes(C_0,x)
            
            x_new = cs.mtimes(A_0,x) + cs.mtimes(B_0,u) + cs.mtimes(A_lpv, 
                    cs.tanh(cs.mtimes(W_A,y))) + cs.mtimes(B_lpv, 
                    cs.tanh(cs.mtimes(W_B,u)))
            y_new = cs.mtimes(C_0,x_new)
            
            
            input = [x,u,A_0,A_lpv,W_A,B_0,B_lpv,W_B,C_0,C_lpv,W_C]
            
            # Remove entries with dimension zero
            input = [i for i in input if all(i.shape)]
            input_names = [var.name() for var in input]
            
            output = [x_new,y_new]
            output_names = ['x_new','y_new']  
            
            self.Function = cs.Function(name, input, output, input_names,
                                        output_names)

            # Calculate affine parameters
            theta_A = cs.tanh(cs.mtimes(W_A,y))/cs.mtimes(W_A,y)
            # theta_B = XXX
            # theta_C = XXX
            
            # theta = cs.vertcat(theta_A,theta_B,theta_C)   
            
            # self.AffineParameters = cs.Function('AffineParameters',input,
            #                                     [theta],input_names,['theta'])

            self.ParameterInitialization()
            
            return None


class LinearSSM(LPV_RNN):
    """
    
    """

    def __init__(self,dim_u,dim_x,dim_y,u_lab,y_lab,initial_params=None, 
                 frozen_params = [], init_proc='random'):
        
        self.dim_u = dim_u
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.name = 'LinearSSM'
        self.dim = dim_x
        
        self.u_label = u_lab
        self.y_label = y_lab
        
        
        self.InitialParameters = initial_params
        self.InitializationProcedure = init_proc
        self.FrozenParameters = frozen_params
        
        self.Initialize(initial_params)

    def Initialize(self,initial_params=None):
            
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
            D = cs.MX.sym('D',dim_y,dim_u)
            
            # Put all Parameters in Dictionary with random initialization
            self.Parameters = {'A':np.random.rand(dim_x,dim_x),
                               'B':np.random.rand(dim_x,dim_u),
                               'C':np.random.rand(dim_y,dim_x),
                               'D':np.random.rand(dim_y,dim_u)}
        
            # self.Input = {'u':np.random.rand(u.shape)}
            
            # Define Model Equations
            x_new = cs.mtimes(A,x) + cs.mtimes(B,u)
            y_new = cs.mtimes(C,x_new) + cs.mtimes(D,u) 
            
            
            # Remove entries with dimension zero
            input = [i for i in input if all(i.shape)]
            input_names = [var.name() for var in input]
            
            output = [x_new,y_new]
            output_names = ['x_new','y_new']  
            
            self.Function = cs.Function(name, input, output, input_names,output_names)
            
            self.ParameterInitialization()
            
            return None
   

class GRU():
    """
    Modell des Bauteils, welches die einwirkenden Prozessgrößen auf die 
    resultierenden Bauteilqualität abbildet.    
    """

    def __init__(self,dim_u,dim_c,dim_hidden,dim_out,name):
        
        self.dim_u = dim_u
        self.dim_c = dim_c
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.name = name
        
        self.Initialize()  
 

    def Initialize(self):
        
        dim_u = self.dim_u
        dim_c = self.dim_c
        dim_hidden = self.dim_hidden
        dim_out = self.dim_out
        name = self.name      
        
        u = cs.MX.sym('u',dim_u,1)
        c = cs.MX.sym('c',dim_c,1)
        
        # Parameters
        # RNN part
        W_r = cs.MX.sym('W_r',dim_c,dim_u+dim_c)
        b_r = cs.MX.sym('b_r',dim_c,1)
    
        W_z = cs.MX.sym('W_z',dim_c,dim_u+dim_c)
        b_z = cs.MX.sym('b_z',dim_c,1)    
        
        W_c = cs.MX.sym('W_c',dim_c,dim_u+dim_c)
        b_c = cs.MX.sym('b_c',dim_c,1)    
    
        # MLP part
        W_h = cs.MX.sym('W_z',dim_hidden,dim_c)
        b_h = cs.MX.sym('b_z',dim_hidden,1)    
        
        W_o = cs.MX.sym('W_c',dim_out,dim_hidden)
        b_o = cs.MX.sym('b_c',dim_out,1)  
        
        # Put all Parameters in Dictionary with random initialization
        self.Parameters = {'W_r':np.random.rand(W_r.shape[0],W_r.shape[1]),
                           'b_r':np.random.rand(b_r.shape[0],b_r.shape[1]),
                           'W_z':np.random.rand(W_z.shape[0],W_z.shape[1]),
                           'b_z':np.random.rand(b_z.shape[0],b_z.shape[1]),
                           'W_c':np.random.rand(W_c.shape[0],W_c.shape[1]),
                           'b_c':np.random.rand(b_c.shape[0],b_c.shape[1]),                          
                           'W_h':np.random.rand(W_h.shape[0],W_h.shape[1]),
                           'b_h':np.random.rand(b_h.shape[0],b_h.shape[1]),                           
                           'W_o':np.random.rand(W_o.shape[0],W_o.shape[1]),
                           'b_o':np.random.rand(b_o.shape[0],b_o.shape[1])}
        
        # Equations
        f_r = logistic(cs.mtimes(W_r,cs.vertcat(u,c))+b_r)
        f_z = logistic(cs.mtimes(W_z,cs.vertcat(u,c))+b_z)
        
        c_r = f_r*c
        
        f_c = cs.tanh(cs.mtimes(W_c,cs.vertcat(u,c_r))+b_c)
        
        
        c_new = f_z*c+(1-f_z)*f_c
        
        h =  cs.tanh(cs.mtimes(W_h,c_new)+b_h)
        x_new = cs.mtimes(W_o,h)+b_o    
    
        
        # Casadi Function
        # Remove entries with dimension zero
        input = [i for i in input if all(i.shape)]
        input_names = [var.name() for var in input]
        
        output = [c_new,x_new]
        output_names = ['c_new','x_new']
    
        self.Function = cs.Function(name, input, output, input_names,output_names)

        return None
    
    
class SilverBoxPhysikal():
    """
    
    """

    def __init__(self,initial_params=None,name=None):
        
        self.name = name
        
        self.Initialize()

    def Initialize(self,initial_params=None):
            
        # For convenience of notation
        name = self.name
        
        # Define input, state and output vector
        u = cs.MX.sym('u',1,1)
        x = cs.MX.sym('x',2,1)
        y = cs.MX.sym('y',1,1)
        
        # Define Model Parameters
        dt = 1/610.352    #1/610.35  cs.MX.sym('dt',1,1)  #Sampling rate fixed from literature
        d = cs.MX.sym('d',1,1)
        a = cs.MX.sym('a',1,1)
        b = cs.MX.sym('b',1,1)
        m = cs.MX.sym('m',1,1)
        
        
        m_init = 1.e-05*abs(np.random.rand(1,1))#1.09992821e-05
        d_init = 2*m_init*21.25
        a_init = d_init**2/(4*m_init)+437.091**2*m_init
        # dt_init = np.array([[ 1/610.35]])
        
        # Put all Parameters in Dictionary with random initialization
        self.Parameters = {'d':d_init,#0.01+0.001*np.random.rand(1,1),
                           'a':a_init,#2+0.001*np.random.rand(1,1),
                           'b':0.01*abs(np.random.rand(1,1)),
                           'm':m_init}#0.0001+0.001*np.random.rand(1,1)}
        
       
    
        # continuous dynamics
        x_new = cs.vertcat(x[1],(-(a + b*x[0]**2)*x[0] - d*x[1] + u)/m)
        
        input = [x,u,d,a,b,m]
        input_names = ['x','u','d','a','b','m']
        
        output = [x_new]
        output_names = ['x_new']  
        
        
        f_cont = cs.Function(name,input,output,
                             input_names,output_names)  
        
        x1 = RK4(f_cont,input,dt)
        
        C = np.array([[1,0]])
        y1 = cs.mtimes(C,x1)
        
        self.Function = cs.Function(name, input, [x1,y1],
                                    input_names,['x1','y1'])

        # Calculate affine parameters
        theta = -(x[0]**2)
        self.AffineParameters = cs.Function('AffineParameters',input,
                                            [theta],input_names,['theta'])
            
        return None
   
    
    def AffineStateSpaceMatrices(self,x0,u0,params=None):
        
        # A = self.Function.jacobian('x1','x')
        # B = self.Function.jacobian('x1','u')
        # C = self.Function.jacobian('x1','u')

        J = self.Function.jacobian()
        # B = self.Function.jacobian()
        # C = self.Function.jacobian()        

        if params==None:
            params = self.Parameters
        
        params_new = []
            
        for name in  self.Function.name_in():
            try:
                params_new.append(params[name])                      # Parameters are already in the right order as expected by Casadi Function
            except:
                continue
        
        # Must be provided to jacobian, value completely arbitrary
        x_out = np.array([[0],[0]])
        y_out = np.array([[0]])
        
        J = J(x0,u0,*params_new,x_out,y_out)
        A = np.array(J[0:2,0:2])
        B = np.array(J[0:2,2])
        C = np.array([[1, 0]])
        
        
        return A,B,C



class MLP():
    """
    Implementation of a single-layered Feedforward Neural Network.
    """

    def __init__(self,dim_u,dim_x,dim_hidden,name):
        """
        Initialization procedure of the Feedforward Neural Network Architecture
        
        
        Parameters
        ----------
        dim_u : int
            Dimension of the input, e.g. dim_u = 2 if input is a 2x1 vector
        dim_x : int
            Dimension of the state, e.g. dim_x = 3 if state is a 3x1 vector.
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
        self.dim_x = dim_x
        self.name = name
        
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
        dim_x = self.dim_x 
        name = self.name
    
        u = cs.MX.sym('u',dim_u,1)
        x = cs.MX.sym('x',dim_x,1)
        
        # Model Parameters
        W_h = cs.MX.sym('W_h',dim_hidden,dim_u+dim_x)
        b_h = cs.MX.sym('b_h',dim_hidden,1)
        
        W_o = cs.MX.sym('W_out',dim_x,dim_hidden)
        b_o = cs.MX.sym('b_out',dim_x,1)
        
        # Put all Parameters in Dictionary with random initialization
        self.Parameters = {'W_h':np.random.rand(W_h.shape[0],W_h.shape[1]),
                           'b_h':np.random.rand(b_h.shape[0],b_h.shape[1]),
                           'W_o':np.random.rand(W_o.shape[0],W_o.shape[1]),
                           'b_o':np.random.rand(b_o.shape[0],b_o.shape[1])}
    
       
        # Model Equations
        h =  cs.tanh(cs.mtimes(W_h,cs.vertcat(u,x))+b_h)
        x_new = cs.mtimes(W_o,h)+b_o
        
        
        input = [x,u,W_h,b_h,W_o,b_o]
        input_names = ['x','u','W_h','b_h','W_o','b_o']
        
        output = [x_new]
        output_names = ['x_new']  
        
        self.Function = cs.Function(name, input, output, input_names,output_names)
        
        return None
   
    def OneStepPrediction(self,x0,u0,params=None):

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
        """
        Repeated call of self.OneStepPrediction() for a given input trajectory
        

        Parameters
        ----------
        x0 : array-like with dimension [self.dim_x, 1]
            initial state resp
        u : array-like with dimension [N,self.dim_u]
            trajectory of input signal with length N
        params : dictionary, optional
            see self.OneStepPrediction()

        Returns
        -------
        x : array-like with dimension [N+1,self.dim_x]
            trajectory of output signal with length N+1 
            
        """
        
        x = []

        # initial states
        x.append(x0)
                      
        # Simulate Model
        for k in range(u.shape[0]):
            x.append(self.OneStepPrediction(x[k],u[[k],:],params))
        
        # Concatenate list to casadiMX
        x = cs.hcat(x).T 
       
        return x


class Rehmer_NN_LPV(Recurrent):
    """
    Quasi-LPV model structure for system identification. Matrices associated
    with the time varying parameter vector supposed to be sparse and only have 
    non-zero coefficients where necessary to approximate the nonlinearity. Each 
    time varying parameter is modelled as a Neural Network. Scheduling variables
    are the states and the input.
    """

    def __init__(self,dim_u,dim_x,dim_y,u_label,y_label,name,dim_thetaA=0,
                 dim_thetaB=0,dim_thetaC=0,NN_A_dim=[],NN_B_dim=[],NN_C_dim=[],
                 NN_A_act=[],NN_B_act=[], NN_C_act=[], initial_params=None,
                 frozen_params = [], init_proc='random'):
        '''
        Initializes the model structure by Rehmer et al. 2021.
        dim_u: int, dimension of the input vector
        dim_x: int, dimension of the state vector
        dim_y: int, dimension of the output vector
        dim_thetaA: int, dimension of the affine parameter associated with the 
        system matrix
        dim_thetaB: int, dimension of the affine parameter associated with the 
        input matrix
        dim_thetaC: int, dimension of the affine parameter associated with the 
        output matrix
        NN_A_dim: list, each entry is a list of integers specifying the number
        of neurons in the layers of the NN 
        NN_B_dim: list, each entry is a list of integers specifying the number
        of neurons in the layers of the NN     
        NN_C_dim: list, each entry is a list of integers specifying the number
        of neurons in the layers of the NN     
        NN_A_act: list, each entry is a list of integers, that specifies the
        activation function used in the layers of the corresponding NN
        NN_B_act: list, each entry is a list of integers, that specifies the
        activation function used in the layers of the corresponding NN
        NN_C_act: list, each entry is a list of integers, that specifies the
        activation function used in the layers of the corresponding NN
        

        initial_params: dict, dictionary specifying the inital parameter values
        name: str, specifies name of the model
        '''       
        
        Recurrent.__init__(self, dim_u, dim_x, dim_y, u_label, y_label, name, 
                           initial_params, frozen_params, init_proc)
        

        self.dim_thetaA = dim_thetaA
        self.dim_thetaB = dim_thetaB
        self.dim_thetaC = dim_thetaC
        
        self.dim = dim_thetaA+dim_thetaB+dim_thetaC
        
        self.NN_A_dim = NN_A_dim
        self.NN_B_dim = NN_B_dim
        self.NN_C_dim = NN_C_dim
        
        self.NN_A_act = NN_A_act
        self.NN_B_act = NN_B_act
        self.NN_C_act = NN_C_act
        
        
        self.Initialize()

    def Initialize(self):
            
        # For convenience of notation
        dim_u = self.dim_u
        dim_x = self.dim_c 
        dim_y = self.dim_out   
        dim_thetaA = self.dim_thetaA
        dim_thetaB = self.dim_thetaB
        dim_thetaC = self.dim_thetaC
        NN_A_dim = self.NN_A_dim
        NN_B_dim = self.NN_B_dim
        NN_C_dim = self.NN_C_dim    
        NN_A_act = self.NN_A_act
        NN_B_act = self.NN_B_act
        NN_C_act = self.NN_C_act
       
        name = self.name
        
        # Define input, state and output vector
        u = cs.MX.sym('u',dim_u,1)
        x = cs.MX.sym('x',dim_x,1)
        
        # Define Model Parameters for the linear part
        A = cs.MX.sym('A',dim_x,dim_x)
        B = cs.MX.sym('B',dim_x,dim_u)
        C = cs.MX.sym('C',dim_y,dim_x)
        
        # Define Model Parameters for the time varying part
        A_i = cs.MX.sym('A_i',dim_x,dim_x,dim_thetaA)
        B_i = cs.MX.sym('B_i',dim_x,dim_u,dim_thetaB)
        C_i = cs.MX.sym('C_i',dim_y,dim_x,dim_thetaC)
        
        input = [x,u,A,B,C,*A_i,*B_i,*C_i]
       
        # Define Parameters of all Neural Networks modelling the affine parameters
        NN_Ai = [[] for i in range(0,dim_thetaA)]
        NN_Bi = [[] for i in range(0,dim_thetaB)]
        NN_Ci = [[] for i in range(0,dim_thetaC)]
        
        NN_Ai_names = ['NN_A'+str(i) for i in range(0,dim_thetaA)]
        NN_Bi_names = ['NN_B'+str(i) for i in range(0,dim_thetaB)]
        NN_Ci_names = ['NN_C'+str(i) for i in range(0,dim_thetaC)]        
        
        
        NN_input_names = []
        # NN_1_dim.append(dim_thetaA)
        for NN, NN_name, NN_dim in zip([*NN_Ai,*NN_Bi,*NN_Ci],
                                       [*NN_Ai_names,*NN_Bi_names,*NN_Ci_names],
                                       [*NN_A_dim,*NN_B_dim,*NN_C_dim]):
            
            for l in range(0,len(NN_dim)):
            
                if l == 0:
                    params = [cs.MX.sym(NN_name+'_Wx'+str(l),NN_dim[l],dim_x),
                              cs.MX.sym(NN_name+'_Wu'+str(l),NN_dim[l],dim_u),
                              cs.MX.sym(NN_name+'_b'+str(l),NN_dim[l],1)]
                else:
                    params = [cs.MX.sym(NN_name+'_W'+str(l),NN_dim[l],NN_dim[l-1]),
                              cs.MX.sym(NN_name+'_b'+str(l),NN_dim[l],1)]

                input.extend(params)
                
                NN.append(params)
        
        
        # Define Model Equations
       
        # Calculate the activations of the NNs by looping over each NN and
        # each layer
        NN_out = [[0] for i in range(dim_thetaA+dim_thetaB+dim_thetaC)]
        
        for out,NN,NN_act in zip(NN_out,
                                 [*NN_Ai,*NN_Bi,*NN_Ci],
                                 [*NN_A_act,*NN_B_act,*NN_C_act]):
            
            out.extend(Eval_FeedForward_NN(cs.vertcat(x,u),NN,NN_act))
        
        # Multiply  matrices and NN outputs
        
        NN_mult = []
        
        for matrix,out in zip([*A_i,*B_i,*C_i],NN_out):
            NN_mult.append(matrix*out[-1])
        
        
        A_nl = sum(NN_mult[0:dim_thetaA])
        B_nl = sum(NN_mult[dim_thetaA:dim_thetaA+dim_thetaB])
        C_nl = sum(NN_mult[dim_thetaA+dim_thetaB::])
        
        # State and output equation
        x_new = cs.mtimes(A+A_nl,x) + cs.mtimes(B+B_nl,u)
        y_new = cs.mtimes(C+C_nl,x_new)
        
        
        # Define inputs and outputs for casadi function
        
        # Remove entries with dimension zero
        input = [i for i in input if all(i.shape)]
        input_names = [var.name() for var in input]
        
        # NN_Ai_flat = [p for NN in NN_Ai for layer in NN for p in layer]
        # NN_Bi_flat = [p for NN in NN_Bi for layer in NN for p in layer]
        # NN_Ci_flat = [p for NN in NN_Ci for layer in NN for p in layer]
        
        # input = [x,u,A,B,C,*A_i,*B_i,*C_i,*NN_Ai_flat,*NN_Bi_flat,*NN_Ci_flat]
        
        # A_i_names = ['A_'+str(i) for i in range(0,dim_thetaA)]
        # B_i_names = ['B_'+str(i) for i in range(0,dim_thetaB)]
        # C_i_names = ['C_'+str(i) for i in range(0,dim_thetaC)]
        
        # input_names = ['x','u','A','B','C',*A_i_names,*B_i_names,
        #                 *C_i_names,*NN_input_names]
      
        output = [x_new,y_new]
        output_names = ['x_new','y_new']
        
        self.Function = cs.Function(name, input, output, input_names,
                                    output_names)
       
        # Calculate affine parameters
        # theta_A = NN_out[0][-1] * cs.tanh(cs.mtimes(E_1,x))/cs.mtimes(E_1,x)
        # theta_B = NN_out[1][-1] * cs.tanh(cs.mtimes(E_2,u))/cs.mtimes(E_2,u)
        # theta_C = NN_out[2][-1] * cs.tanh(cs.mtimes(E_3,x))/cs.mtimes(E_3,x)
        
        # theta = cs.vertcat(theta_A,theta_B,theta_C)   
        
        # self.AffineParameters = cs.Function('AffineParameters',input,
        #                                     [theta],input_names,['theta'])
        
        # Initialize symbolic variables with numeric values
        self.ParameterInitialization()
        
        return None
    

    
    
    
    
class DummySystem1(LPV_RNN):
    """
    
    """

    def __init__(self,dim_u,dim_x,dim_y,u_lab,y_lab,initial_params=None, 
                 frozen_params = [], init_proc='random'):
        
        self.dim_u = dim_u
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.name = 'DummySystem1'
        self.dim = dim_x
        
        self.u_label = u_lab
        self.y_label = y_lab
        
        
        self.InitialParameters = initial_params
        self.InitializationProcedure = init_proc
        self.FrozenParameters = frozen_params
        
        self.Initialize(initial_params)

    def Initialize(self,initial_params=None):
            
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
            x_new = cs.tanh(cs.mtimes(A,x)) + cs.mtimes(B,u) 
            y_new = cs.mtimes(C,x_new)
            
            dfdx = cs.jacobian(x_new,x)
            dfdu = cs.jacobian(x_new,u)
            dgdx = cs.jacobian(y_new,x)
            
            
            input = [x,u,A,B,C]
            input_names = ['x','u','A','B','C']
            
            output = [x_new,y_new]
            output_names = ['x_new','y_new']  
            
            self.Function = cs.Function(name, input, output, input_names,output_names)
            
            self.ParameterInitialization()
            
            return None
        
        
class DummySystem2(LPV_RNN):
    """
    
    """

    def __init__(self,dim_u,dim_x,dim_y,dim_h,u_lab,y_lab,initial_params=None, 
                 frozen_params = [], init_proc='random'):
        
        self.dim_u = dim_u
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_h = dim_h
        
        self.name = 'DummySystem2'

        
        self.u_label = u_lab
        self.y_label = y_lab
        
        
        self.InitialParameters = initial_params
        self.InitializationProcedure = init_proc
        self.FrozenParameters = frozen_params
        
        self.Initialize(initial_params)

    def Initialize(self,initial_params=None):
            
            # For convenience of notation
            dim_u = self.dim_u
            dim_x = self.dim_x 
            dim_y = self.dim_y     
            dim_h = self.dim_h
            
            name = self.name
            
            # Define input, state and output vector
            u = cs.MX.sym('u',dim_u,1)
            x = cs.MX.sym('x',dim_x,1)
            y = cs.MX.sym('y',dim_y,1)
            
            # Define Model Parameters
            A = cs.MX.sym('A',dim_x,dim_x)
            B = cs.MX.sym('B',dim_x,dim_u)
            C = cs.MX.sym('C',dim_y,dim_x)
            W_h = cs.MX.sym('W_h',dim_y,dim_x)
            b_h = cs.MX.sym('b_h',dim_h,1)
            W_o = cs.MX.sym('W_o',dim_y,dim_h)
            b_o = cs.MX.sym('b_o',dim_y,1)
            
            # Put all Parameters in Dictionary with random initialization
            self.Parameters = {'A':np.random.rand(dim_x,dim_x),
                               'B':np.random.rand(dim_x,dim_u),
                               'C':np.random.rand(dim_y,dim_x),
                               'W_h': np.random.rand(dim_y,dim_x),
                               'b_h': np.random.rand(dim_h,1),
                               'W_o': np.random.rand(dim_y,dim_h),
                               'b_o': np.random.rand(dim_y,1)}
        
            # self.Input = {'u':np.random.rand(u.shape)}
            
            # Define Model Equations
            NN = cs.mtimes(W_o,cs.tanh(cs.mtimes(W_h,x)+b_h))+b_o
            x_new = cs.mtimes(A,x) + cs.mtimes(B,u)+ NN
            y_new = cs.mtimes(C,x_new)
            
            y_old = cs.mtimes(C,x)
            
            dfdx = cs.jacobian(x_new,x)
            dfdu = cs.jacobian(x_new,u)
            dgdx = cs.jacobian(y_old,x)
            
            input = [x,u,A,B,C,W_h,b_h,W_o,b_o]
            input_names = ['x','u','A','B','C','W_h','b_h','W_o','b_o']

            # output = [x_new,y_new]
            # output_names = ['x_new','y_new']              

            output = [x_new,y_new,y_old,dfdx,dfdu,dgdx]
            output_names = ['x_new','y_new','y_old','dfdx','dfdu','dgdx']  
            
            self.Function = cs.Function(name, input, output, input_names,output_names)
            
            self.ParameterInitialization()
            
            return None
        
class LPV_NARX(LPV_RNN):
    """

    """

    def __init__(self,dim_u,dim_y,shifts, dim_theta, initial_params=None, 
                 frozen_params = [], init_proc='random'):
        
        self.dim_u = dim_u
        self.shifts = shifts
        self.dim_y = dim_y
        self.dim_theta = dim_theta
        self.dim = dim_theta
        
        self.InitialParameters = initial_params
        self.FrozenParameters = frozen_params
        self.InitializationProcedure = init_proc
        
        self.name = 'LPV_NARX'
        
        self.Initialize()

    def Initialize(self):
            
        # For convenience of notation
        dim_u = self.dim_u
        dim_y = self.dim_y   
        shifts = self.shifts
        dim_theta = self.dim_theta


        # Define inputs
        u = cs.MX.sym('u',dim_u,shifts)
        # theta = cs.MX.sym('theta',dim_theta,shifts)
        y_in = cs.MX.sym('y',dim_y,shifts)
        
        # Define Model Parameters
        alpha = cs.MX.sym('alpha',dim_y,dim_y*dim_theta,shifts)
        beta = cs.MX.sym('beta',dim_y,dim_u*dim_theta,shifts)
        
        if dim_theta == 1:
            theta = cs.vertcat(cs.power(y_in,0))
        elif dim_theta == 2:
            theta = cs.vertcat(cs.power(y_in,0),
                           cs.power(y_in,1))
               
        # Define Model Equations, loop over all local models
        y_new = 0
        
        for l in range(0,shifts):
            
            a = 0
            b = 0
            
            for p in range(0,dim_theta):
                a = a + cs.mtimes(alpha[l][:,p*dim_y:(p+1)*dim_y],
                                 theta[p,l])   
                b = b + cs.mtimes(beta[l][:,p*dim_u:(p+1)*dim_u],
                                 theta[p,l])
                
                                                           
            y_new = y_new + cs.mtimes(a,y_in[:,l])  +   cs.mtimes(b,u[:,l]) 

        
        # Define Casadi Function
       
        # Define input of Casadi Function and save all parameters in 
        # dictionary
                    
        input = [y_in,u]
        input_names = ['y_in','u']
        
        Parameters = {}
        
        # Add local model parameters
        for l in range(0,shifts):
            input.extend([alpha[l],beta[l]])    
            input_names.extend(['alpha'+str(l),'beta'+str(l)])
            
            
         
        output = [y_new]
        output_names = ['y_new']  
        
        self.Function = cs.Function(self.name, input, output, input_names,
                                    output_names)

      
        self.ParameterInitialization()
        
        return None
    
    
    def OneStepPrediction(self,y_in,u,params=None):
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
            
        y_new = self.Function(y_in,u,*params_new)     
                              
        return y_new

    def Simulation(self,y0,u,params=None):
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
        # y.append(np.flip(y0))
        
        y0 = y0.reshape((self.shifts,self.dim_y)).T
        # u = u.reshape((-1,self.dim_u,self.shifts))
        
        x.append(y0)

                      
        # Simulate Model
        for k in range(u.shape[0]):
            
            U = u[k,:].reshape((self.shifts,self.dim_u)).T
            
            y_new = self.OneStepPrediction(y0,U,params)

            
            
            y0 = cs.horzcat(y_new,y0[:,0:-1])
            
            x.append(y0)
            y.append(y_new)
        
        # Concatenate list to casadiMX
        y = cs.hcat(y).T    
        x = cs.hcat(x).T
       
        return x,y  