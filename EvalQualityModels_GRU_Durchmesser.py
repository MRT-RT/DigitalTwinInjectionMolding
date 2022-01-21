# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 16:45:55 2021

@author: alexa
"""
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from DIM.models.model_structures import GRU
from DIM.models.injection_molding import QualityModel
from DIM.optim.common import BestFitRate
from DIM.miscellaneous.PreProcessing import arrange_data_for_ident, eliminate_outliers, LoadData


def Eval_GRU_on_Val(charges,counter,path):
    
    dim_c = 2
      
    u_inj_lab= ['p_wkz_ist','T_wkz_ist','p_inj_ist','Q_Vol_ist','V_Screw_ist']
    u_press_lab = ['p_wkz_ist','T_wkz_ist','p_inj_ist','Q_Vol_ist','V_Screw_ist']
    u_cool_lab = ['p_wkz_ist','T_wkz_ist','p_inj_ist','Q_Vol_ist','V_Screw_ist']
    
    y_lab = ['Durchmesser_innen']
    
    u_lab= [u_inj_lab] #,u_press_lab,u_cool_lab]
    
    # Load best model
    res = pkl.load(open(path+'GRU_Durchmesser_innen_c'+str(counter)+'.pkl','rb'))
       
    params = res.loc[res['loss_val'].idxmin()][['params']][0]
    
    # Load data
    data,cycles_train_label,cycles_val_label,charge_train_label,charge_val_label = \
    LoadData(dim_c,charges,y_lab,u_lab)
    
    # Initialize model structures
    injection_model = GRU(dim_u=5,dim_c=dim_c,dim_hidden=10,dim_out=1,name='inject')
    # press_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=10,dim_out=1,name='press')
    # cool_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=10,dim_out=1,name='cool')
 
    quality_model = QualityModel(subsystems=[injection_model],
                                  name='q_model_Durchmesser_innen')
    
    # Assign best parameters to model
    quality_model.AssignParameters(params)
    
    # Evaluate model on training data
    y_train = []
    e_train = []
    y_train_hist = []
    c_train_hist = []

    for i in range(len(data['u_train'])): 
        c,y = quality_model.Simulation(data['init_state_train'][i], data['u_train'][i],None,data['switch_train'][i])
        
        c = np.array(c)
        y = np.array(y)
        
        y_train_hist.append(y)
        c_train_hist.append(c)
        
        y_train.append(y[-1][0])
        e_train.append(data['y_train'][i]-y_train[-1])
    
    
    y_true = np.array(data['y_train']).reshape((-1,1))
    y_train = np.array(y_train).reshape((-1,1))
    e_train = np.array(e_train).reshape((-1,1))
    cycles_train_label = np.array(cycles_train_label).reshape((-1,))
    charge_train_label = np.array(charge_train_label).reshape((-1,1))  
    
    results_train = pd.DataFrame(data=np.hstack([y_true,y_train,e_train,
                                  charge_train_label]),
                                index = cycles_train_label,
                            columns=['y_true','y_est','e','charge'])
    
    # Evaluate model on validation data
    y_val = []
    e_val = []
    y_val_hist = []
    c_val_hist = []
    
    for i in range(len(data['u_val'])): 
        c,y = quality_model.Simulation(data['init_state_val'][i], data['u_val'][i],None,data['switch_val'][i])
        
        c = np.array(c)
        y = np.array(y)
        
        y_val_hist.append(y)
        c_val_hist.append(c)
        
        y_val.append(y[-1][0])
        e_val.append(data['y_val'][i]-y_val[-1])
    
    
    y_true = np.array(data['y_val']).reshape((-1,1))
    y_val = np.array(y_val).reshape((-1,1))
    e_val = np.array(e_val).reshape((-1,1))
    cycles_val_label = np.array(cycles_val_label).reshape((-1,))
    charge_val_label = np.array(charge_val_label).reshape((-1,1))
    
    results_val = pd.DataFrame(data=np.hstack([y_true,y_val,e_val,
                                  charge_val_label]),
                                index = cycles_val_label,
                            columns=['y_true','y_est','e','charge'])


    return results_train, results_val, data, quality_model


Modellierungsplan = pkl.load(open('Modellierungsplan.pkl','rb'))
Modellierungsplan = Modellierungsplan[0:12]
counter = range(1,12)

path = 'Results/GRU_2c_1sub_5in_Plan_c1_c14/'

for charges,c in  zip(Modellierungsplan,counter):
    results_train, results_val, data, quality_model = Eval_GRU_on_Val(charges,c,path)
    
    pkl.dump(results_train,open(path+'GRU_results_train_c'+str(c)+'.pkl','wb')) 
    pkl.dump(results_val,open(path+'GRU_results_val_c'+str(c)+'.pkl','wb')) 
    pkl.dump(quality_model,open(path+'GRU_quality_model_c'+str(c)+'.pkl','wb'))
    pkl.dump(data,open(path+'data_c'+str(c)+'.pkl','wb'))