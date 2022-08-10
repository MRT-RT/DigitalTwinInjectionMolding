# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 15:16:22 2022

@author: LocalAdmin
"""

import sys
sys.path.insert(0, '/home/alexander/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, 'E:/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, 'C:/Users/LocalAdmin/Documents/GitHub/DigitalTwinInjectionMolding/')

from DIM.miscellaneous.PreProcessing import LoadFeatureData, MinMaxScale
from DIM.optim.common import BestFitRate
from DIM.models.model_structures import Static_Multi_MLP
from DIM.optim.param_optim import ParallelModelTraining, static_mode

import multiprocessing

import pickle as pkl
import numpy as np
import pandas as pd


path_sys = 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/'
# path_sys = 'C:/Users/LocalAdmin/Documents/GitHub/DigitalTwinInjectionMolding/'
# path_sys = '/home/alexander/GitHub/DigitalTwinInjectionMolding/' 
# path_sys = 'E:/GitHub/DigitalTwinInjectionMolding/'

path = path_sys + '/data/Versuchsplan/normalized/'

def Eval_MLP(dim_hidden,init,charges,path):
    
    res = pkl.load(open('QM_Di_MLP_'+str(dim_hidden)+'.pkl','rb'))
   
    # params = res.loc[res['loss_val'].idxmin()][['params_val']][0]
    
    params = res.loc[init][['params_val']][0]   
    
    split = 'all'
    del_outl = True
    
    data_train,data_val = LoadFeatureData(path,charges,split,del_outl)
    
    u_label = ['Düsentemperatur', 'Werkzeugtemperatur',
                'Einspritzgeschwindigkeit', 'Umschaltpunkt', 'Nachdruckhöhe',
                'Nachdruckzeit', 'Staudruck', 'Kühlzeit','T_wkz_0','p_inj_0',
                'x_0']
    
    
    y_label = ['Durchmesser_innen']   
    
    # Normalize Data
    data_train,minmax = MinMaxScale(data_train,u_label+y_label)
    data_val,_ = MinMaxScale(data_val,u_label+y_label,minmax)
    
    model = Static_Multi_MLP(dim_u=11, dim_out=1, dim_hidden=dim_hidden,
                             layers=2, u_label=u_label, y_label=y_label,
                             name='MLP', init_proc='xavier')
    

    # Assign best parameters to model
    model.Parameters = params
    
    # Evaluate model on training data
    _,y_train = static_mode(model,data_train)
    y_true = data_train[y_label].values.reshape((-1,1))
    e_train = y_true-y_train
    
    results_train = pd.DataFrame(data=np.hstack([y_true,y_train,e_train]),
                            columns=['y_true','y_est','e'],
                              index = data_train.index)
    
    # Evaluate model on validation data
    _,y_val = static_mode(model,data_val)
        
    y_true = data_val[y_label].values.reshape((-1,1))
    e_val = y_true-y_val
    
    results_val = pd.DataFrame(data=np.hstack([y_true,y_val,e_val]),
                            columns=['y_true','y_est','e'],
                            index = data_val.index)

    return results_train,results_val

# Find charges with high / low variance
plan = pkl.load(open(path+'Versuchsplan.pkl','rb'))

charges_low_var = []
charges_high_var = []


for charge in list(range(1,275)):
    plan_sub = plan.loc[plan['Charge']==charge]
    
    std = plan_sub['Durchmesser_innen'].std()
    
    if std > 0.035:
        charges_high_var.append(charge)
    else:
        charges_low_var.append(charge)

data = []

for c in range(1,11):

    for init in range(0,20):    

        results_train,results_val = Eval_MLP(c,init,charges_low_var,
                                                    path)
        
        BFR = BestFitRate(results_val['y_true'].values.reshape((-1,1)),
              results_val['y_est'].values.reshape((-1,1)))/100
        
        print('dim c:'+str(c)+' init:' + str(init) + ' BFR: ' + 
              str(BFR))
        
        data.append([BFR,'MLP_2l_set_x0',c,'Durchmesser_innen',init])
        
df = pd.DataFrame(data=data,columns=['BFR','model','complexity','target','init'])

pkl.dump(df,open('MLP_set_x0_Durchmesser_low_var.pkl','wb'))





