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

from DIM.models.model_structures import GRU
from DIM.models.injection_molding import QualityModel
from DIM.optim.common import BestFitRate
from DIM.optim.param_optim import parallel_mode
from DIM.miscellaneous.PreProcessing import arrange_data_for_ident, eliminate_outliers, LoadDynamicData


def Eval_GRU_on_Val(dim_c):

    # Load best model
    res = pkl.load(open('GRU_c'+str(dim_c)+'_3sub_Stoergrsn_Gewicht.pkl','rb'))
       
    params = res.loc[res['loss_val'].idxmin()][['params_val']][0]
    # params = res.loc[10]['params_val']

    charges = list(range(1,26))
    
    mode='quality'
    split = 'all'
    
    del_outl = False
    # split = 'part'
    
    # path_sys = 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/'  
    path_sys = '/home/alexander/GitHub/DigitalTwinInjectionMolding/'
    # path_sys = 'E:/GitHub/DigitalTwinInjectionMolding/'
    
    path_data_train = 'data/Stoergroessen/20220504/Versuchsplan/normalized/'
    
    path_data_strgrsn = 'data/Stoergroessen/20220506/Rezyklat_Stoerung/normalized/'
    # path_data_strgrsn = 'data/Stoergroessen/20220504/Umschaltpkt_Stoerung/normalized/'
    # path_data_strgrsn = 'data/Stoergroessen/20220505/T_wkz_Stoerung/normalized/'
    
    
    u_inj= ['p_wkz_ist','T_wkz_ist']
    u_press= ['p_wkz_ist','T_wkz_ist']
    u_cool= ['p_wkz_ist','T_wkz_ist']
    
    u_lab = [u_inj,u_press,u_cool]
    y_lab = ['Gewicht']

   

    data_train,data_val = LoadDynamicData(path_sys+path_data_train,charges,
                                          split,y_lab,u_lab,mode,True)
    
    data_st1,data_st2 = LoadDynamicData(path_sys+path_data_strgrsn,[1],
                                        split, y_lab,u_lab,mode,False)
    
    data_st = data_st1 # pd.concat([data_st1,data_st2])
    
    c0_train = [np.zeros((dim_c,1)) for i in range(0,len(data_train['data']))]
    c0_st = [np.zeros((dim_c,1)) for i in range(0,len(data_st['data']))] 
    
    data_train['init_state'] = c0_train
    data_st['init_state'] = c0_st
    
    
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
    
    # # Evaluate model on training data
    # _,y_train = parallel_mode(quality_model,data_train)
    
    
    # y_true = np.array([df[y_lab].iloc[0] for df in data_train['data']]).reshape((-1,1))
    # y_train = np.array([df[y_lab].iloc[0] for df in y_train]).reshape((-1,1))
    # e_train = y_true-y_train
    
    # results_train = pd.DataFrame(data=np.hstack([y_true,y_train,e_train]),
    #                         columns=['y_true','y_est','e'],
    #                           index = data_train['cycle_num'])
    
    results_train = None
    
    # Evaluate model on validation data
    _,y_val = parallel_mode(quality_model,data_st)
    
    y_true = np.array([df[y_lab].iloc[0] for df in data_st['data']]).reshape((-1,1))
    y_val = np.array([df[y_lab].iloc[0] for df in y_val]).reshape((-1,1))
    e_val = y_true-y_val
    
    
    results_st = pd.DataFrame(data=np.hstack([y_true,y_val,e_val]),
                            columns=['y_true','y_est','e'],
                            index = data_st['cycle_num'])

    return results_train,results_st


for i in range(4,5):

    results_train,results_st = Eval_GRU_on_Val(dim_c=i)
    
    print(BestFitRate(results_st['y_true'].values.reshape((-1,1)),
                results_st['y_est'].values.reshape((-1,1))))

    plt.figure()
    sns.stripplot(x = results_st.index, y=abs(results_st['y_true']-results_st['y_est']))
    # plt.ylim([-0.02,0.02])
    plt.title(str(i))
    
    
fig,ax = plt.subplots(1,1)

weight_c11 = 7.991


sns.stripplot(x = results_st.index, y=results_st['y_true']+weight_c11-1,ax=ax,
              color='grey')
# sns.stripplot(x = results_st.index, y=results_st['y_est'],ax=ax)

# ax.set_xlabel('$n$')
# ax.set_ylabel('$\mathrm{BFR}$')


plt.vlines(x=[results_st.index.get_loc(loc) for loc in [20,45,65,85,105]], 
           ymin=0, ymax=10, colors='k', linestyles='dashed')

plt.hlines(y=[8.093,8.178], 
           xmin=0, xmax=130, colors='grey', linestyles='dashed')

xticks = [0,3] + list(np.arange(8,58+5,5)) + [62,67,72,76,81,86,90,95,100,
                                              105,109,114,118]
xticks = xticks[0::2]
ax.set_xticks(ax.get_xticks()[xticks])

ax.set_ylabel('m')
ax.set_xlabel('c')


ax.set_ylim([1.08+weight_c11-1,1.22+weight_c11-1])
ax.set_xlim([0,120])
fig.set_size_inches((20/2.54,6/2.54))
plt.tight_layout()

plt.savefig('Data_Disturbance_Recyc.png', bbox_inches='tight',dpi=600)    

    # sns.stripplot(x = results_st.index, y=results_st['y_true'],color='grey')
    # sns.stripplot(x = results_st.index, y=results_st['y_est'])
    # plt.ylim([-0.02,0.02])
    # plt.title(str(i))
    

# plt.plot(results_val['y_true'],'o')
# plt.plot(results_val['y_est'],'o')
# plt.plot(results_val['e'],'o')

# plt.plot(results_st['y_true'],'o')
# plt.plot(results_st['y_est'],'o')
# plt.plot(results_st['y_true'], results_st['e'],'o')

# pkl.dump(results_train,open('GRU_results_train_c'+str(c)+'.pkl','wb')) 
# pkl.dump(results_val,open('GRU_results_val_c'+str(c)+'.pkl','wb')) 
# pkl.dump(quality_model,open('GRU_quality_model_c'+str(c)+'.pkl','wb'))
# pkl.dump(data,open('data_c'+str(c)+'.pkl','wb'))