#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 09:42:06 2022

@author: alexander
"""
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from DIM.miscellaneous.PreProcessing import MinMaxScale

from DIM.optim.common import BestFitRate

from DIM.optim.param_optim import static_mode

def Eval_MLP(model,data_train,data_val,minmax):
    
    data_train_norm,_ = MinMaxScale(data_train,minmax=minmax)
    data_val_norm,_ = MinMaxScale(data_val,minmax=minmax)      
    
    # Evaluate model on training data
    _,pred = static_mode(model,data_train_norm)
    
    # Unnormalize prediction
    pred = MinMaxScale(pred,minmax=minmax,reverse=True)
    
    true = data_train[model.y_label]
    
    error = true-pred
    
    BFR = BestFitRate(true,pred)/100
    
    pred.rename(columns={lab:lab+'_est' for lab in model.y_label},inplace = True)
    error.rename(columns={lab:lab+'_error' for lab in model.y_label},inplace = True)
    
    results_train = {'pred':pd.concat([pred,true,error],axis=1),
                     'BFR':BFR}
        
    # Evaluate model on validation data
    _,pred = static_mode(model,data_val_norm)
    
    # Unnormalize prediction
    pred = MinMaxScale(pred,minmax=minmax,reverse=True)
    
    true = data_val[model.y_label]
    
    error = true-pred
    
    BFR = BestFitRate(true,pred)/100
    
    pred.rename(columns={lab:lab+'_est' for lab in model.y_label},inplace = True)
    error.rename(columns={lab:lab+'_error' for lab in model.y_label},inplace = True)
    
    results_val = {'pred':pd.concat([pred,true,error],axis=1),
                     'BFR':BFR}

    return results_train,results_val

def cycles_dates(hdf5):
    
    index = []
    
    for cycle in hdf5.keys():
        cycle_num = int(hdf5[cycle]['f071_Value']['block0_values'][:])
        index.append(cycle_num)
    
    df = pd.DataFrame(index=index,data=[],columns=['Date'])
    
    hdf5.close()
    
    return df


def estimate_polynomial(degree,inputs,targets,data_train,data_val):
    
    
    n_u = len(inputs)
    n_y = len(targets)
    
    
    # Generate Feature Transformer
    poly = PolynomialFeatures(degree)
    
    # Transform input data (training and validation) appropriately
    X_train = poly.fit_transform(data_train[inputs])
    X_val = poly.transform(data_val[inputs])    
    
    # Initialize polynomial model and fit to training data
    PolyModel = LinearRegression()
    PolyModel.fit(X_train,data_train[targets])
    
    # print(PolyModel.score(X_train,data_train[targets]))
    
    
    # Evaluate model on training and validation data
    

    results = []
    
    y_est_lab = [y+'_est' for y in targets] 
    e_lab = [y+'_error' for y in targets]
    columns = targets + y_est_lab + e_lab
    
    for data in [data_train,data_val]:
        
        X = poly.transform(data[inputs])   
        
        y_est = PolyModel.predict(X)
        y_true = data[targets].values
        
        e = y_true-y_est
    
        results.append(pd.DataFrame(data=np.hstack([y_true,y_est,e]),
                                columns=columns,
                                index = data.index))
        
        
    
    e_train =  np.linalg.norm(results[0][e_lab])
    e_val = np.linalg.norm(results[1][e_lab])
    
    
    BFR_train = BestFitRate(results[0][targets],results[0][y_est_lab])
    BFR_val = BestFitRate(results[1][targets],results[1][y_est_lab])
    
    
    return {'model': PolyModel,
            'BFR_train': BFR_train,
            'e_train': e_train,
            'predict_train': results[0],
            'BFR_val': BFR_val,
            'predict_val': results[1],
            'e_val': e_val}
    
