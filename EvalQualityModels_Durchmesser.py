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


def Eval_GRU_on_Val(charges,counter):
    
    dim_c = 2
    path = 'Results/17_01_2022/'
    # Load best model
    res = pkl.load(open(path+'GRU_Durchmesser_innen_c'+str(counter)+'.pkl','rb'))
       
    params = res.loc[res['loss_val'].idxmin()][['params']][0]
    
    # Load data
    data,cycles_train_label,cycles_val_label,charge_train_label,charge_val_label = \
    LoadData(dim_c,charges)
    
    # Initialize model structures
    injection_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=10,dim_out=1,name='inject')
    press_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=10,dim_out=1,name='press')
    cool_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=10,dim_out=1,name='cool')
 
    quality_model = QualityModel(subsystems=[injection_model,press_model,cool_model],
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


    return results_train, results_val


res = []
Modellierungsplan = pkl.load(open('Modellierungsplan.pkl','rb'))

for i in range(0,1):
    res.append(Eval_GRU_on_Val(Modellierungsplan[i],i+1))

BFR = np.array(BFR)

results_train, results_val = Eval_GRU_on_Val(Modellierungsplan[12],13)
 


plt.figure()
sns.stripplot(x="charge", y="e", data=results_train,
              size=4, color=".3", linewidth=0)
sns.stripplot(x="charge", y="e", data=results_val,
              size=4,  linewidth=0)


plt.figure()
sns.stripplot(x=results_train.index, y="y_true", data=results_train,
              size=4, color=".3", linewidth=0)
sns.stripplot(x=results_train.index, y="y_est", data=results_train,
              size=4,  linewidth=0)


plt.figure()
plt.plot(results_train['y_true'],results_train['y_est'],'o')

plt.figure()
plt.hist(results_train['e'].values,bins=40)
plt.xlabel(['error'])

plt.figure()
plt.plot(results_train['cycle'],results_train['y_true'],'o')

# plt.plot(np.array(data['y_val']),np.array(y_val),'o')
# plt.xlim([27.2,27.9])
# plt.ylim([27.2,27.9])

# plt.figure()
# plt.hist(np.array(e_val),bins=40)
# plt.xlabel(['error'])

# plt.figure()
# plt.plot(np.array(data['y_val']),np.array(e_val),'o')
# plt.xlabel(['y_true'])
# plt.ylabel(['error'])


# plt.figure()
# sns.stripplot(x="charge", y="e", data=results_train,
#               size=4, color=".3", linewidth=0)
# sns.stripplot(x="charge", y="e", data=results_val,
#               size=4,  linewidth=0)



# # Charge 1
# idx_train = cycles_train_label[np.where(charge_train_label[:] == 1)[0]].reshape((-1,))
# idx_val = cycles_val_label[np.where(charge_val_label == 1)[0]].reshape((-1,))

# plt.figure()
# plt.plot(results_train[results_train['charge']==1]['y_true'],'d',markersize=12,label='train true')
# plt.plot(results_train[results_train['charge']==1]['y_est'],'d',markersize=12,label='train est')
# # plt.ylim([27.5,27.9])

# plt.figure()
# plt.plot(results_train[results_train['charge']==1]['y_true'],'d',markersize=12,label='val true')
# plt.plot(results_train[results_train['charge']==1]['y_est'],'d',markersize=12,label='val est')
# plt.ylim([27.5,27.9])



# plt.subplot(2,3,6,title='D innen',xlabel='cycle')
# plt.plot([2,3,4,5,6],q_train,'d',markersize=12,label='train true')
# plt.plot([7,8,9],q_val,'d',markersize=12,label='val true')
# plt.legend()
# plt.subplots_adjust(hspace=0.3)
# plt.show()

'''
TO DO:
- Residuen gruppiert nach Faktorstufen plotten
- Residuen im Histogramm plotten
- Residuen über wahre Zielgröße plotten    
- Woher kommen harte Begrenzungen oben und unten?
'''


# e_val = np.array(e_val).reshape((-1,1))
# plt.hist(e_val)



# u_lab= ['p_wkz_ist','T_wkz_ist']#,'p_inj_ist','Q_Vol_ist','V_Screw_ist']
# plt.close('all')

# plt.figure()

# for i in range(0,1):
    
#     cycle_num = cycles_val_label[i]

#     cycle_data = pkl.load(open('./data/Versuchsplan/cycle'+str(cycle_num)+'.pkl','rb'))
    
#     # plt.figure()
#     # plt.plot(cycle_data[u_lab])
#     # plt.legend(u_lab)
#     # plt.title(str(cycle_num))
    
#     plt.figure()
#     plt.plot(cycle_data.index[0:y_val_hist[i].shape[0]],y_val_hist[i])
#     plt.legend(['Q'])
#     plt.title(str(cycle_num))

#     plt.figure()
#     plt.plot(cycle_data.index[0:c_val_hist[i].shape[0]],c_val_hist[i])
#     plt.legend(['c1','c2','c3','c4','c5','c6','c7'])
#     plt.title(str(cycle_num))
    
    # y_lab = ['Durchmesser_innen']
    # u_inj_lab= ['p_wkz_ist','T_wkz_ist','p_inj_ist','Q_Vol_ist','V_Screw_ist']
    # u_press_lab = u_inj_lab
    # u_cool_lab = ['p_wkz_ist','T_wkz_ist']

















