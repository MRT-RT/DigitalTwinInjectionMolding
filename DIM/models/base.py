# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 12:51:56 2022

@author: alexa
"""

import casadi as cs
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from DIM.optim.common import RK4
from scipy.stats import ortho_group
from DIM.models.initializations import XavierInitialization, RandomInitialization, HeInitialization

# from miscellaneous import *
class Model():
    
    def __init__(self,dim_u,dim_out,u_label,y_label,name,initial_params, 
                 frozen_params, init_proc):
        
        """
        Initialization procedure of the Feedforward Neural Network Architecture
        
        Parameters
        ----------
        dim_u : int
            Dimension of the input, e.g. dim_u = 2 if input is a 2x1 vector
        dim_out : int
            Dimension of the output, e.g. dim_out = 3 if output is a 3x1 vector.
        u_label : 
        y_label : 
        name : str
            Name of the model, e.g. name = 'InjectionPhaseModel'.
        initial_params :
        frozen_params :
        init_proc :    
        
            Returns
        -------
        None.

        """
               
        self.dim_u = dim_u
        self.dim_out = dim_out
        
        self.u_label = u_label
        self.y_label = y_label
        self.name = name
        
        self.initial_params = initial_params
        self.frozen_params = frozen_params
        self.init_proc = init_proc

    def MinMaxScale(self,df,**kwargs):
        
        df = df.copy()
        
        # Check if scaling is to be reversed
        reverse = kwargs.pop('reverse',False)
         
        if reverse:
            
            col_min = self.minmax[0]
            col_max = self.minmax[1]
            
            if all(col_min.keys()==col_max.keys()):
                cols = col_min.keys()
    
            cols = [col for col in cols if col in df.columns]
            
            # Unscale from 0,1
            df_rev = df[cols]* (col_max - col_min) + col_min
            
            # Unscale from -1,1
            # df_rev = 1/2* ( (df[cols] + 1) * (col_max - col_min)) + col_min
            
            
            df.loc[:,cols] = df_rev
        
        # Else scale data
        else:
            
            # Check if column min and max already exist from previous
            # scaling
            try:
                col_min = self.minmax[0]
                col_max = self.minmax[1]
                
                # 
                if all(col_min.keys()==col_max.keys()):
                    cols = col_min.keys()
                
                cols = [col for col in cols if col in df.columns]
                
            except:
                
                # If not calculate  column min and max from given data
                cols = self.u_label + self.y_label
               
                col_min = df[cols].min()
                col_max = df[cols].max()
                
                
                # Check if col_min equals col_max, fix  to avoid division by zero
                for col in cols:
                    if col_min[col] == col_max[col]:
                        col_min[col] = 0.0
            
                # save for scaling of future data
                self.minmax = (col_min,col_max)
                
            # Scale to -1,1
            # df_norm = 2*(df[cols] - col_min) / (col_max - col_min) - 1 
            
            # Scale to 0,1   
            df_norm = (df[cols] - col_min) / (col_max - col_min)
        
            df.loc[:,cols] = df_norm
        
        
        return df

class Static(Model):
    """
    Base implementation of a static model.
    """

    def __init__(self,dim_u,dim_out,u_label,y_label,name,initial_params, 
                 frozen_params, init_proc):
        
        Model.__init__(self,dim_u,dim_out,u_label,y_label,name,initial_params,
                       frozen_params,init_proc)
        
        
    
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
            
        for name in self.Function.name_in()[1::]:
            try:
                params_new.append(params[name])                      # Parameters are already in the right order as expected by Casadi Function
            except:
                params_new.append(self.Parameters[name])
        
        if isinstance(u0,pd.DataFrame):
            u0 = np.array(u0[self.u_label].values,dtype=float)
        
        y = self.Function(u0,*params_new)     
                              
        return y

    def static_mode(self,data,params=None):
        
        
        y_est = []
       
        # u = data[model.u_label].values
        u = data[self.u_label]
        
        # If parameters are not given only calculate model prediction    
        if params is None:
            
            # One-Step prediction
            for k in range(u.shape[0]):  
                # y_new = model.OneStepPrediction(u[k,:],params)
                y_new = self.OneStepPrediction(u.iloc[[k]],params)
                
                y_est.append(y_new)
            
            y_est = np.array(y_est).reshape((-1,len(self.y_label)))
            
            
            # Rename columns to distinguish estimate from true value
            # cols_pred = [label + '_est' for label in self.y_label]
            
            cols = self.y_label
            
            df_pred = pd.DataFrame(data=y_est, columns=cols, index=data.index)
            
            try:
                loss = np.sum((data[self.y_label]-df_pred[self.y_label]).values**2)
            except:
                print('No output data given to calculate loss.')
                loss = None
                    
        # else calulate loss
        else:
            
            y_ref = data[self.y_label].values
            
            loss = 0
            e = [] 
            # One-Step prediction
            for k in range(u.shape[0]):
                y_new = self.OneStepPrediction(u.iloc[[k]],params)
                # y_new = self.OneStepPrediction(u[k,:],params)
                y_est.append(y_new)
                e.append(y_ref[k,:]-y_new)
                loss = loss + cs.sumsqr(e[-1])
                
            df_pred = None
            
        return loss,df_pred
                      
    def AssignParameters(self,params):
        
        for p_name in self.Function.name_in()[1::]:
            try:
                self.Parameters[p_name] = params[p_name]
            except:
                continue


class Recurrent(Model):
    '''
    Parent class for all recurrent Models
    '''

    def __init__(self,dim_u,dim_c,dim_out,u_label,y_label,name,initial_params, 
                  frozen_params, init_proc):
        
        Model.__init__(self,dim_u,dim_out,u_label,y_label,name,initial_params,
                        frozen_params,init_proc)
        
        self.dim_c = dim_c
    
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

    def parallel_mode(self,data,params=None):
          
        loss = 0
    
        simulation = []
        
        # Loop over all batches 
        for i in range(0,len(data['data'])):
            
            io_data = data['data'][i]
            x0 = data['init_state'][i]
            try:
                kwargs = {'switching_instances':data['switch'][i]}            
            except KeyError:
                kwargs = {'switching_instances':None} 
            
            
            u = io_data.iloc[0:-1][self.u_label]
            # u = io_data[self.u_label]
    
            
            # Simulate Model        
            pred = self.Simulation(x0,u,params,**kwargs)
            
            
            y_ref = io_data[self.y_label]
            
            
            if isinstance(pred, tuple):           
                x_est= pred[0]
                y_est= pred[1]
            else:
                y_est = pred
                
            # Calculate simulation error            
            # Check for the case, where only last value is available
            
            if np.all(np.isnan(y_ref[1:])):
                
                y_ref = y_ref.iloc[0].values
                y_est=y_est[-1,:]
                e= y_ref - y_est
                loss = loss + cs.sumsqr(e)
                
                idx = [i]
        
            else :
                
                # y_ref = y_ref[1:1+y_est.shape[0],:]                             # first observation cannot be predicted
                e = y_ref.iloc[1:].values - y_est
                loss = loss + cs.sumsqr(e)
                
                idx = y_ref.iloc[1:].index
            
            if params is None:
                df = pd.DataFrame(data = np.array(y_est),
                                     columns=self.y_label,
                                     index = idx)
                
                simulation.append(df)
            else:
                simulation = None
                
        return loss,simulation

    def series_parallel_mode(self,data,params=None):
      
        loss = 0
        
        x = []
        
        prediction = []
    
        # if None is not None:
            
        #     print('This is not implemented properly!!!')
            
        #     for i in range(0,len(u)):
        #         x_batch = []
        #         y_batch = []
                
        #         # One-Step prediction
        #         for k in range(u[i].shape[0]-1):
                    
                    
                    
        #             x_new,y_new = self.OneStepPrediction(x_ref[i][k,:],u[i,k,:],
        #                                                   params)
        #             x_batch.append(x_new)
        #             y_batch.append(y_new)
                    
        #             loss = loss + cs.sumsqr(y_ref[i][k,:]-y_new) + \
        #                 cs.sumsqr(x_ref[i,k+1,:]-x_new) 
                
        #         x.append(x_batch)
        #         y.append(y_batch)
            
        #     return loss,x,y 
        
        # else:
        for i in range(0,len(data['data'])):
            
            io_data = data['data'][i]
            x0 = data['init_state'][i]
            switch = data['switch'][i]
            
            y_est = []
            # One-Step prediction
            for k in range(0,io_data.shape[0]-1):
                
                uk = io_data.iloc[k][self.u_label].values.reshape((-1,1))
                yk = io_data.iloc[k][self.y_label].values.reshape((-1,1))
                
                ykplus = io_data.iloc[k+1][self.y_label].values.reshape((-1,1))
                
                # predict x1 and y1 from x0 and u0
                y_new = self.OneStepPrediction(yk,uk,params)
                
                loss = loss + cs.sumsqr(ykplus-y_new)        
                
                y_est.append(y_new.T)
            
            y_est = cs.vcat(y_est)
            
            if params is None:
                y_est = np.array(y_est)
                
                df = pd.DataFrame(data=y_est, columns=self.y_label,
                                  index=io_data.index[1::])
                
                prediction.append(df)
            else:
                prediction = None
            
        return loss,prediction
    
    def SetParameters(self,params):
            
        for p_name in self.Function.name_in()[2::]:
            try:
                self.Parameters[p_name] = params[p_name]
            except:
                pass      