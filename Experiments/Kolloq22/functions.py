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

from DIM.optim.common import BestFitRate

def cycles_dates(hdf5):
    
    index = []
    
    for cycle in hdf5.keys():
        cycle_num = int(hdf5[cycle]['f071_Value']['block0_values'][:])
        index.append(cycle_num)
    
    df = pd.DataFrame(index=index,data=[],columns=['Date'])
    
    hdf5.close()
    
    return df


def estimate_polynomial(degree,inputs,targets,data_train,data_val):
    
    # Generate Feature Transformer
    poly = PolynomialFeatures(degree)
    
    # Transform input data (training and validation) appropriately
    X_train = poly.fit_transform(data_train[inputs])
    X_val = poly.transform(data_val[inputs])    
    
    # Initialize polynomial model and fit to training data
    PolyModel = LinearRegression()
    PolyModel.fit(X_train,data_train[targets])
    
    
    
    # Evaluate model on training and validation data
    

    results = []
    
    for data in [data_train,data_val]:
        
        X = poly.transform(data[inputs])   
        
        y_est = PolyModel.predict(X).reshape((-1,1))
        y_true = data[targets[0]].values.reshape((-1,1))
        
        e = y_true-y_est
    
        results.append(pd.DataFrame(data=np.hstack([y_true,y_est,e]),
                                columns=['y_true','y_est','e'],
                                index = data.index))
    
  
    BFR_train = BestFitRate(results[0]['y_true'],results[0]['y_est'])
    BFR_val = BestFitRate(results[1]['y_true'],results[1]['y_est'])
    
    
    return {'model': PolyModel,
            'BFR_train': BFR_train,
            'predict_train': results[0],
            'BFR_val': BFR_val,
            'predict_val': results[1]}
    
