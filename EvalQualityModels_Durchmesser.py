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
from DIM.miscellaneous.PreProcessing import arrange_data_for_ident, eliminate_outliers

# Load Data

def LoadData(dim_c):
    
    # Load Versuchsplan to find cycles that should be considered for modelling
    data = pkl.load(open('./data/Versuchsplan/Versuchsplan.pkl','rb'))
    
    data = eliminate_outliers(data)
    
    # Delete outliers rudimentary
    
    # Cycles for parameter estimation
    cycles_train_label = []
    cycles_val_label = []
    
    charge_train_label = []
    charge_val_label = []
    
    for charge in range(1,274):
        cycles = data[data['Charge']==charge].index.values
        cycles_train_label.append(cycles[-6:-1])
        cycles_val_label.append(cycles[-1])
        
        charge_train_label.extend([charge]*len(cycles[-6:-1]))
        charge_val_label.extend([charge]*len(cycles[[-1]]))
    
    cycles_train_label = np.hstack(cycles_train_label)
    cycles_val_label = np.hstack(cycles_val_label)
    
    # Delete cycles that for some reason don't exist
    charge_train_label = np.delete(charge_train_label, np.where(cycles_train_label == 767)) 
    cycles_train_label = np.delete(cycles_train_label, np.where(cycles_train_label == 767)) 
    
    
    
    # # Load cycle data, check if usable, convert to numpy array
    cycles_train = []
    cycles_val = []
    
    for c in cycles_train_label:
        cycles_train.append(pkl.load(open('data/Versuchsplan/cycle'+str(c)+'.pkl',
                                          'rb')))
    
    for c in cycles_val_label:
        cycles_val.append(pkl.load(open('data/Versuchsplan/cycle'+str(c)+'.pkl',
                                          'rb')))
    
    # Select input and output for dynamic model
    y_lab = ['Durchmesser_innen']
    u_inj_lab= ['p_wkz_ist','T_wkz_ist','p_inj_ist','Q_Vol_ist','V_Screw_ist']
    u_press_lab = u_inj_lab
    u_cool_lab = ['p_wkz_ist','T_wkz_ist']
    # 
    x_train,q_train,switch_train  = arrange_data_for_ident(cycles_train,y_lab,
                                        [u_inj_lab,u_press_lab,u_cool_lab],'quality')
    #
    # x_train,q_train,switch_train = arrange_data_for_qual_ident(cycles_train,x_lab,q_lab)
    
    x_val,q_val,switch_val = arrange_data_for_ident(cycles_val,y_lab,
                                        [u_inj_lab,u_press_lab,u_cool_lab],'quality')
    
    c0_train = [np.zeros((dim_c,1)) for i in range(0,len(x_train))]
    c0_val = [np.zeros((dim_c,1)) for i in range(0,len(x_val))]
    
    data = {'u_train': x_train,
            'y_train': q_train,
            'switch_train': switch_train,
            'init_state_train': c0_train,
            'u_val': x_val,
            'y_val': q_val,
            'switch_val': switch_val,
            'init_state_val': c0_val}
    
    return data,cycles_train_label,cycles_val_label,charge_train_label,charge_val_label

dim_c = 2

data,cycles_train_label,cycles_val_label,charge_train_label,charge_val_label = \
LoadData(dim_c=dim_c)

path = './temp/PSO_param/q_model_Durchmesser_innen/'


# Load results
results = pkl.load(open('GRU_Durchmesser_innen.pkl','rb'))
param = results.loc[0]['params']

# Initialize model structure
injection_model = GRU(dim_u=5,dim_c=dim_c,dim_hidden=10,dim_out=1,name='inject')
press_model = GRU(dim_u=5,dim_c=dim_c,dim_hidden=10,dim_out=1,name='press')
cool_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=10,dim_out=1,name='cool')

quality_model = QualityModel(subsystems=[injection_model,press_model,cool_model],
                              name='q_model_Durchmesser_innen')

quality_model.AssignParameters(param)


# Evaluate model on training and validation data
y_val = []
e_val = []
y_val_hist = []
c_val_hist = []

for i in range(0,273): 
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

# y_train = []
y_train = []
e_train = []
y_train_hist = []
c_train_hist = []

for i in range(0,1362): 
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

# Plot results

plt.plot(np.array(data['y_val']),np.array(y_val),'o')
plt.xlim([27.2,27.9])
plt.ylim([27.2,27.9])

plt.figure()
plt.hist(np.array(e_val),bins=40)
plt.xlabel(['error'])

plt.figure()
plt.plot(np.array(data['y_val']),np.array(e_val),'o')
plt.xlabel(['y_true'])
plt.ylabel(['error'])


plt.figure()
sns.stripplot(x="charge", y="e", data=results_train,
              size=4, color=".3", linewidth=0)
sns.stripplot(x="charge", y="e", data=results_val,
              size=4,  linewidth=0)

plt.figure()
plt.plot(results_train['cycle'],results_train['y_true'],'o')

# Charge 123
idx_train = cycles_train_label[np.where(charge_train_label[:] == 123)[0]].reshape((-1,))
idx_val = cycles_val_label[np.where(charge_val_label == 123)[0]].reshape((-1,))

plt.figure()
plt.plot(results_train[results_train['charge']==123]['y_true'],'d',markersize=12,label='train true')
plt.plot(results_train[results_train['charge']==123]['y_est'],'d',markersize=12,label='train est')
plt.ylim([27.5,27.9])

plt.figure()
plt.plot(results_train[results_train['charge']==204]['y_true'],'d',markersize=12,label='val true')
plt.plot(results_train[results_train['charge']==204]['y_est'],'d',markersize=12,label='val est')
plt.ylim([27.5,27.9])



plt.subplot(2,3,6,title='D innen',xlabel='cycle')
plt.plot([2,3,4,5,6],q_train,'d',markersize=12,label='train true')
plt.plot([7,8,9],q_val,'d',markersize=12,label='val true')
plt.legend()
plt.subplots_adjust(hspace=0.3)
plt.show()

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

















