# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:44:37 2021

@author: alexa
"""

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing

from DIM.miscellaneous.PreProcessing import arrange_data_for_ident, eliminate_outliers

from DIM.models.model_structures import GRU,LSTM
from DIM.models.injection_molding import QualityModel
from DIM.optim.param_optim import ModelTraining, HyperParameterPSO
from DIM.miscellaneous.PreProcessing import LoadData

   
def Fit_LSTM_to_Charges(charges,counter):
    
    path = 'Results/LSTM_2c_1sub_2in_allCharg/'
    dim_c = 2

    u_lab= [['p_wkz_ist','T_wkz_ist']]
    y_lab = ['Durchmesser_innen']
    
    data,cycles_train_label,cycles_val_label,charge_train_label,charge_val_label = \
    LoadData(dim_c,charges,y_lab,u_lab)
    
    one_model = LSTM(dim_u=2,dim_c=dim_c,dim_hidden=10,dim_out=1,name='inject')
    
    for rnn in [one_model]:
        name = rnn.name
        
        initial_params = {'b_f_'+name: np.random.uniform(0,2,(dim_c,1)),
                          'b_i_'+name: np.random.uniform(-2,0,(dim_c,1)),
                          'b_o_'+name: np.random.uniform(-2,0,(dim_c,1))}
        
        rnn.InitialParameters = initial_params
        
    quality_model = QualityModel(subsystems=[one_model],
                                  name='q_model_Durchmesser_innen')
    
    
    s_opts = {"hessian_approximation": 'limited-memory',"max_iter": 3000,
              "print_level":2}
    
    
    results_LSTM = ModelTraining(quality_model,data,initializations=50, BFR=False, 
                      p_opts=None, s_opts=s_opts)
    
    results_LSTM['Chargen'] = 'c'+str(counter)
    
    pkl.dump(results_LSTM,open(path+'LSTM_Durchmesser_innen_c'+str(counter)+'.pkl','wb'))

    return results_LSTM  


    
if __name__ == '__main__':
    
    print('Process started..')
    

        
    Modellierungsplan = pkl.load(open('Modellierungsplan_all.pkl','rb'))
    counter = [411]
    
    multiprocessing.freeze_support()
    
    pool = multiprocessing.Pool()
    
    result = pool.starmap(Fit_LSTM_to_Charges, zip(Modellierungsplan,counter) ) 
