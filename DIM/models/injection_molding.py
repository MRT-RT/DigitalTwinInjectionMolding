# -*- coding: utf-8 -*-

from sys import path
path.append(r"C:\Users\LocalAdmin\Documents\casadi-windows-py38-v3.5.5-64bit")

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np

# from miscellaneous import *


class ProcessModel():
    '''
    Container for the model which estimates the quality of the part given 
    trajectories of the process variables
    '''
    def __init__(self,subsystems,name):
        """
        Initialization routine for the QualityModel class. 
    
        Parameters
        ----------
        subsystems : list,
             list of models, each model describing a disctinct phase of the
             injection molding process 
        name : string, name of the model
    
        Returns
        -------    
        """

        # self.reference = []
        # self.ref_params = {}       
        # self.subsystems = []
        # self.switching_instances = []
              
        self.subsystems = subsystems
        self.switching_instances = []
        self.name = name
        self.FrozenParameters = []
        
        dim_out = []
        u_label = []
        y_label = []

        
        for subsystem in self.subsystems:
            dim_out.append(subsystem.dim_out)
            u_label.extend(subsystem.u_label)
            y_label.extend(subsystem.y_label)
            
            object.__setattr__(self, 'dim_u'+'_'+subsystem.name, 
                                subsystem.dim_u)
            object.__setattr__(self, 'dim_hidden'+'_'+subsystem.name, 
                                subsystem.dim_hidden)
        
        # Check consistency      
        if sum(dim_out)/len(dim_out)==dim_out[0]:
            self.dim_out = dim_out[0]
        else:
            raise ValueError('State dimension of all subsystems needs to be equal')
        self.u_label = list(set(u_label))
        self.y_label = list(set(y_label))
        
        self.Initialize()
    
    def Initialize(self):
        """
        Re-Initializes each of the subsystems according to its own 
        initialization routine. Model structure parameters for re-initialization
        are taken from the attributes of the QualityModel instance. This
        routine is called during multi-start parameter optimization when random
        initialization of the subsystems is necessary.
        
        Parameters
        ----------
    
        Returns
        ----
        """
       
        # Update attributes of each subsystem
        for subsystem in self.subsystems:
            
            setattr(subsystem, 'dim_u', 
                    object.__getattribute__(self,'dim_u'+'_'+subsystem.name))
            setattr(subsystem, 'dim_hidden', 
                    object.__getattribute__(self,'dim_hidden'+'_'+subsystem.name))
            setattr(subsystem, 'dim_out',
                    object.__getattribute__(self,'dim_out'))
            
            # Call Initialize function of each subsystem
            subsystem.Initialize()
            
        self.ParameterInitialization()
        
        return None
    
    def Simulation(self,x0,u,params=None,switching_instances=None,**kwargs):
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
        
        
        
        if self.switching_instances is not None:
            switching_instances = [u.index.get_loc(s) for s in self.switching_instances]
            
            switching_instances = [0] + switching_instances + [len(u)]
            # switching_instances = [0] + switching_instances + [u.index[-1]]
            u_switched = []
            
            for s in range(len(switching_instances)-1):
                
                u_switched.append(u.iloc[switching_instances[s]:switching_instances[s+1]])
                # u_switched.append(u.loc[switching_instances[s]:switching_instances[s+1]].values)
                # u_switched.append(u.loc[switching_instances[s]:switching_instances[s+1]])
            u = u_switched
        
        # Create empty arrays for output y and hidden state c
        y = []
        x = []   
        
        # System Dynamics as Path Constraints
        for system,u_sys in zip(self.subsystems,u):
            
            # Do a one step prediction based on the model
            sim = system.Simulation(x0,u_sys,params)
            
            # OneStepPrediction can return a tuple, i.e. state and output. The 
            # output is per convention the second entry
            if isinstance(sim,tuple):
                x.append(sim[0])
                y.append(sim[1])    
                
                # Last hidden state is inital state for next model
                x0 = sim[0][-1,:].T

        y = cs.vcat(y)  
        x = cs.vcat(x)          
            
        return x,y  

    
    def ParameterInitialization(self):
        
        self.Parameters = {}
        self.FrozenParameters = []
        
        for system in self.subsystems:
            system.ParameterInitialization()
            self.Parameters.update(system.Parameters)                                  # append subsystems parameters
            self.FrozenParameters.extend(system.FrozenParameters)

    def SetParameters(self,params):
        
        self.Parameters = {}
        
        for system in self.subsystems:
            system.SetParameters(params)
            self.Parameters.update(system.Parameters)
            
        
class QualityModel():
    '''
    Container for the model which estimates the quality of the part given 
    trajectories of the process variables
    '''
    def __init__(self,subsystems,name):
        """
        Initialization routine for the QualityModel class. 
    
        Parameters
        ----------
        subsystems : list,
             list of models, each model describing a disctinct phase of the
             injection molding process 
        name : string, name of the model
    
        Returns
        -------    
        """
              
        self.subsystems = subsystems
        self.switching_instances = []
        self.name = name
        self.FrozenParameters = []
        
        dim_c = []
        dim_out = []
        u_label = []
        y_label = []
        
        for subsystem in self.subsystems:
            dim_c.append(subsystem.dim_c)
            dim_out.append(subsystem.dim_out)
            
            u_label.extend(subsystem.u_label)
            y_label.extend(subsystem.y_label)
            
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
            raise ValueError('Dimension of output all subsystems needs to be equal')
            
        self.u_label = list(set(u_label))
        self.y_label = list(set(y_label))
            
        self.Initialize()

    def Initialize(self):
        """
        Re-Initializes each of the subsystems according to its own 
        initialization routine. Model structure parameters for re-initialization
        are taken from the attributes of the QualityModel instance. This
        routine is called during multi-start parameter optimization when random
        initialization of the subsystems is necessary.
        
        Parameters
        ----------
    
        Returns
        ----
        """
       
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
            
        self.ParameterInitialization()
        
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
        
        
        
        if self.switching_instances is not None:
            switching_instances = [u.index.get_loc(s) for s in self.switching_instances]
            
            switching_instances = [0] + switching_instances + [len(u)]
            # switching_instances = [0] + switching_instances + [u.index[-1]]
            u_switched = []
            
            for s in range(len(switching_instances)-1):
                
                u_switched.append(u.iloc[switching_instances[s]:switching_instances[s+1]])
                # u_switched.append(u.loc[switching_instances[s]:switching_instances[s+1]].values)
                # u_switched.append(u.loc[switching_instances[s]:switching_instances[s+1]])
            u = u_switched
        
        
        # Create empty arrays for output y and hidden state c
        y = []
        c = []   
        
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

    def SetParameters(self,params):
        
        self.Parameters = {}
        
        for system in self.subsystems:
            system.SetParameters(params)
            self.Parameters.update(system.Parameters)

