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
import time

import sys
sys.path.insert(0, "E:\GitHub\DigitalTwinInjectionMolding")
sys.path.insert(0, '/home/alexander/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, 'C:/Users/LocalAdmin/Documents/GitHub/DigitalTwinInjectionMolding/')

from DIM.models.model_structures import GRU
from DIM.models.injection_molding import QualityModel
from DIM.optim.common import BestFitRate
from DIM.optim.param_optim import parallel_mode
from DIM.miscellaneous.PreProcessing import arrange_data_for_ident, eliminate_outliers, LoadDynamicData

# path_sys = 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/'
# path_sys = 'C:/Users/LocalAdmin/Documents/GitHub/DigitalTwinInjectionMolding/'
path_sys = '/home/alexander/GitHub/DigitalTwinInjectionMolding/' 
# path_sys = 'E:/GitHub/DigitalTwinInjectionMolding/'

path = path_sys + '/data/Versuchsplan/normalized/'


def Eval_GRU_on_Val(dim_c,init, charges,path):

    # Load best model
    res = pkl.load(open('GRU_c'+str(dim_c)+'_3sub_all.pkl','rb'))
       
    # params = res.loc[res['loss_val'].idxmin()]['params_val']
    params = res.loc[init]['params_val']
    
    mode='quality'
    split = 'all'
    # split = 'part'
    del_outl = True
          
   
    u_inj= ['p_wkz_ist','T_wkz_ist']
    u_press= ['p_wkz_ist','T_wkz_ist']
    u_cool= ['p_wkz_ist','T_wkz_ist']
    
    u_lab = [u_inj,u_press,u_cool]
    y_lab = ['Durchmesser_innen']

    data_train,data_val = LoadDynamicData(path,charges,split,y_lab,u_lab,
                                          mode,del_outl)
    
    c0_train = [np.zeros((dim_c,1)) for i in range(0,len(data_train['data']))]
    c0_val = [np.zeros((dim_c,1)) for i in range(0,len(data_val['data']))] 
    
    data_train['init_state'] = c0_train
    data_val['init_state'] = c0_val
    
    
    inj_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=1,
                    u_label=u_inj,y_label=y_lab,dim_out=1,name='inj')
    
    press_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=1,
                      u_label=u_press,y_label=y_lab,dim_out=1,name='press')
    
    cool_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=10,
                     u_label=u_cool,y_label=y_lab,dim_out=1,name='cool')
      
    quality_model = QualityModel(subsystems=[inj_model,press_model,cool_model],
                                  name='q_model')    
    
    # Assign best parameters to model
    quality_model.SetParameters(params)
    
    # Evaluate model on training data
    # _,y_train = parallel_mode(quality_model,data_train)
    
    
    # y_true = np.array([df[y_lab].iloc[0] for df in data_train['data']]).reshape((-1,1))
    # y_train = np.array([df[y_lab].iloc[0] for df in y_train]).reshape((-1,1))
    # e_train = y_true-y_train
    
    
    
    # results_train = pd.DataFrame(data=np.hstack([y_true,y_train,e_train]),
    #                         columns=['y_true','y_est','e'])
    results_train = None
    # # Evaluate model on validation data
    _,y_val = parallel_mode(quality_model,data_val)
    
    
    y_true = np.array([df[y_lab].iloc[0] for df in data_val['data']]).reshape((-1,1))
    y_val = np.array([df[y_lab].iloc[0] for df in y_val]).reshape((-1,1))
    e_val = y_true-y_val
    
    
    
    results_val = pd.DataFrame(data=np.hstack([y_true,y_val,e_val]),
                            columns=['y_true','y_est','e'])

    return results_train,results_val


# charges = list(range(1,275))

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

        results_train,results_val = Eval_GRU_on_Val(c,init,charges_high_var,
                                                    path)
        
        BFR = BestFitRate(results_val['y_true'].values.reshape((-1,1)),
              results_val['y_est'].values.reshape((-1,1)))
        
        print('dim c:'+str(c)+' init:' + str(init) + ' BFR: ' + 
              str(BFR))
        
        data.append([BFR,'GRU_3sub',c,'Durchmesser_innen',init])
        
df = pd.DataFrame(data=data,columns=['BFR','model','complexity','target','init'])

pkl.dump(df,open('GRU_3sub_Durchmesser_innen_high_var.pkl','wb'))