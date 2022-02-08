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

from DIM.models.model_structures import Static_MLP
from DIM.models.injection_molding import QualityModel
from DIM.optim.common import BestFitRate
from DIM.miscellaneous.PreProcessing import arrange_data_for_ident, eliminate_outliers, LoadStaticData

from DIM.optim.param_optim import static_mode


def Eval_MLP(charges,dim_hidden):
    
    targets = ['Durchmesser_innen']
    
    path = '/home/alexander/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
    
    data_train,data_val,cycles_train_label,cycles_val_label,\
        charge_train_label,charge_val_label = \
            LoadStaticData(path,charges,targets)
    
    # Normalize Data
    data_max = data_train.max()
    data_min = data_train.min()
    
    data_train = 2*(data_train - data_min)/(data_max-data_min) - 1
    data_val = 2*(data_val - data_min)/(data_max-data_min) - 1
    
    inputs = [col for col in data_train.columns if col not in targets]
    inputs = inputs[0:8]
    
    data = {}
    data['u_train'] = [data_train[inputs].values]
    data['u_val'] = [data_val[inputs].values]
    data['y_train'] = [data_train[targets].values]
    data['y_val'] = [data_val[targets].values]
    
    # Load best model
    res = pkl.load(open('MLP_Durchmesser_innen_dimhidden'+str(dim_hidden)+'.pkl','rb'))   
    params = res.loc[res['loss_val'].idxmin()][['params']][0]
    
    
    # Initialize model structures
    model = Static_MLP(dim_u=8, dim_out=1, dim_hidden=dim_hidden,name='MLP',
                       init_proc='xavier')
    
    # Assign best parameters to model
    model.Parameters = params
    
    # Evaluate model on training data
    y_train = []
    e_train = []
    
    loss_train, e_train,y_train = static_mode(model,data['u_train'],data['y_train'])
   
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
    
    loss_val, e_val,y_val = static_mode(model,data['u_val'],data['y_val'])
    
    
    y_true = np.array(data['y_val']).reshape((-1,1))
    y_val = np.array(y_val).reshape((-1,1))
    e_val = np.array(e_val).reshape((-1,1))
    cycles_val_label = np.array(cycles_val_label).reshape((-1,))
    charge_val_label = np.array(charge_val_label).reshape((-1,1))
    
    results_val = pd.DataFrame(data=np.hstack([y_true,y_val,e_val,
                                  charge_val_label]),
                                index = cycles_val_label,
                            columns=['y_true','y_est','e','charge'])


    return results_train, results_val, data, model


charges = list(range(1,275))

for c in range(1,12):
    results_train, results_val, data, quality_model = Eval_MLP(charges,c)

    pkl.dump(results_train,open('results_train_c'+str(c)+'.pkl','wb')) 
    pkl.dump(results_val,open('results_val_c'+str(c)+'.pkl','wb')) 
    pkl.dump(quality_model,open('quality_model_c'+str(c)+'.pkl','wb'))
    pkl.dump(data,open('data_c'+str(c)+'.pkl','wb'))