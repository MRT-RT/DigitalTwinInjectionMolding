# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:44:37 2021

@author: alexa
"""

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import multiprocessing

from DIM.miscellaneous.PreProcessing import arrange_data_for_ident, eliminate_outliers

from DIM.models.model_structures import GRU,LSTM
from DIM.models.injection_molding import QualityModel
from DIM.optim.param_optim import ModelTraining, HyperParameterPSO
from DIM.miscellaneous.PreProcessing import LoadData


    
def Fit_GRU_to_Charges(charges,counter):
    
    dim_c = 2
    
    data,cycles_train_label,cycles_val_label,charge_train_label,charge_val_label = \
    LoadData(dim_c,charges)
    
    injection_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=10,dim_out=1,name='inject')
    press_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=10,dim_out=1,name='press')
    cool_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=10,dim_out=1,name='cool')
    
    for rnn in [injection_model,press_model,cool_model]:
        name = rnn.name
        
        initial_params = {'b_r_'+name: np.random.uniform(-2,0,(dim_c,1)),
                          'b_z_'+name: np.random.uniform(-1,0,(dim_c,1)),
                          'b_c_'+name: np.random.uniform(-2,0,(dim_c,1))}
        
        rnn.InitialParameters = initial_params
        
    quality_model = QualityModel(subsystems=[injection_model,press_model,cool_model],
                                  name='q_model_Durchmesser_innen')
    
    
    s_opts = {"hessian_approximation": 'limited-memory',"max_iter": 3000,
              "print_level":2}
    
    
    results_GRU = ModelTraining(quality_model,data,initializations=20, BFR=False, 
                      p_opts=None, s_opts=s_opts)
    
    results_GRU['Chargen'] = 'c'+str(counter)
    
    pkl.dump(results_GRU,open('GRU_Durchmesser_innen_c'+str(counter)+'.pkl','wb'))

    return results_GRU  


    
if __name__ == '__main__':
    
    print('Process started..')
    
    

    Modellierungsplan = pkl.load(open('Modellierungsplan.pkl','rb'))
    counter = range(1,14)
    
    multiprocessing.freeze_support()
    
    pool = multiprocessing.Pool()
    
    result = pool.starmap(Fit_GRU_to_Charges, zip(Modellierungsplan,counter) ) 



# idx_min = results_LSTM['loss_val'].idxmin()

# param = results_LSTM.loc[idx_min]['params']

# quality_model.AssignParameters(param)


# # Evaluate model on training and validation data
# y_val = []
# e_val = []
# y_val_hist = []
# c_val_hist = []

# for i in range(0,len(data['y_val'])): 
#     c,y = quality_model.Simulation(data['init_state_val'][i], data['u_val'][i],None,data['switch_val'][i])
    
#     c = np.array(c)
#     y = np.array(y)
    
#     y_val_hist.append(y)
#     c_val_hist.append(c)
    
#     y_val.append(y[-1][0])
#     e_val.append(data['y_val'][i]-y_val[-1])


# y_true = np.array(data['y_val']).reshape((-1,1))
# y_val = np.array(y_val).reshape((-1,1))
# e_val = np.array(e_val).reshape((-1,1))
# cycles_val_label = np.array(cycles_val_label).reshape((-1,))
# charge_val_label = np.array(charge_val_label).reshape((-1,1))

# results_val = pd.DataFrame(data=np.hstack([y_true,y_val,e_val,
#                              charge_val_label]),
#                            index = cycles_val_label,
#                        columns=['y_true','y_est','e','charge'])

# # y_train = []
# y_train = []
# e_train = []
# y_train_hist = []
# c_train_hist = []

# for i in range(0,len(data['y_train'])): 
#     c,y = quality_model.Simulation(data['init_state_train'][i], data['u_train'][i],None,data['switch_train'][i])
    
#     c = np.array(c)
#     y = np.array(y)
    
#     y_train_hist.append(y)
#     c_train_hist.append(c)
    
#     y_train.append(y[-1][0])
#     e_train.append(data['y_train'][i]-y_train[-1])


# y_true = np.array(data['y_train']).reshape((-1,1))
# y_train = np.array(y_train).reshape((-1,1))
# e_train = np.array(e_train).reshape((-1,1))
# cycles_train_label = np.array(cycles_train_label,dtype=int).reshape((-1,))
# charge_train_label = np.array(charge_train_label,dtype=int).reshape((-1,1))

# results_train = pd.DataFrame(data=np.hstack([y_true,y_train,e_train,
#                              charge_train_label]),
#                              index = cycles_train_label,
#                              columns=['y_true','y_est','e','charge'])

# # Plot results
# plt.figure()
# plt.plot(np.array(results_train['y_true']),results_train['y_est'],'o')

# plt.figure()
# plt.plot(np.array(results_train['y_true']),results_train['e'],'o')

# plt.figure()
# plt.plot(np.array(results_train.index),results_train['y_true'],'o')
# plt.plot(np.array(results_train.index),results_train['y_est'],'d')
# plt.plot(np.array(results_val.index),results_val['y_true'],'o')
# plt.plot(np.array(results_val.index),results_val['y_est'],'d')

# plt.figure()
# sns.stripplot(x="charge", y="e", data=results_train,
#               size=4, color=".3", linewidth=0)
# sns.stripplot(x="charge", y="e", data=results_val,
#               size=4,  linewidth=0)

# plt.figure()
# plt.plot(results_train.index,results_train['e'],'d')


#