#!/usr/bin/env python
# coding: utf-8

# # Identification of Quality Models from data

# In[126]:


import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

import sys


sys.path.append('C:/Users/LocalAdmin/Documents/GitHub/DigitalTwinInjectionMolding/')
#sys.path.append('E:/GitHub/DigitalTwinInjectionMolding/')
#sys.path.append('/home/alexander/GitHub/DigitalTwinInjectionMolding/')


from DIM.miscellaneous.PreProcessing import arrange_data_for_ident

from DIM.models.model_structures import GRU,LSTM
from DIM.models.injection_molding import QualityModel
from DIM.optim.param_optim import ModelTraining


# Load experimental data, use 5 cycles for parameter estimation (model training) and 3 cycles for model validation 

# In[127]:


cycles = []

for i in range(1,11):
    cycles.append(pkl.load(open('./data/Versuchsplan/cycle'+str(i)+'.pkl','rb')))

cycles_train = cycles[2:7]
cycles_val = cycles[7:10]


# In[128]:


# print(cycles[0])


# Select process variables and product quality measurements for modeling

# In[129]:


q_lab = ['Durchmesser_innen']
x_lab= ['p_wkz_ist','T_wkz_ist','p_inj_ist','Q_Vol_ist','V_Screw_ist']


# In[130]:


x_train,q_train,switch_train = arrange_data_for_ident(cycles_train,q_lab,
                                                      [x_lab],'quality')

x_val,q_val,switch_val = arrange_data_for_ident(cycles_val,q_lab,
                                                      [x_lab],'quality')

# Plot training data

# In[131]:


# fig, axs = plt.subplots(2,3)
# fig.set_size_inches((40/2.54,20/2.54))
# plt.subplot(2,3,1,title='cycle2',xlabel='k');plt.plot(x_train[0])
# plt.subplot(2,3,2,title='cycle3',xlabel='k');plt.plot(x_train[0])
# plt.subplot(2,3,3,title='cycle4',xlabel='k');plt.plot(x_train[0])
# plt.subplot(2,3,4,title='cycle5',xlabel='k');plt.plot(x_train[0])
# plt.subplot(2,3,5,title='cycle6',xlabel='k');plt.plot(x_train[0])

# plt.subplot(2,3,6,title='D innen',xlabel='cycle')
# plt.plot([2,3,4,5,6],q_train,'d',markersize=12,label='train true')
# plt.plot([7,8,9],q_val,'d',markersize=12,label='val true')
# plt.legend()
# plt.subplots_adjust(hspace=0.3)
# plt.show()


# Initialize Quality Model, comprising one model for the injection, pressure and cooling phase, respectively


dim_c = 2

all_model = GRU(dim_u=5,dim_c=dim_c,dim_hidden=10,dim_out=1,name='inject')

quality_model = QualityModel(subsystems=[all_model],
                              name='q_model')



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

# Usually the model and the data is given to the ModelTraining() procedure, to estimate the optimal parameters, this has been done in advance so the results are merely loaded by calling pkl.load()

# results = ModelTraining(quality_model,data,3)
# results = pkl.load(open('./QualityModel_GRU_one_2c_5in_1out.pkl','rb'))
# print(results)


# The parameters the led to the best results on the validation dataset (row index 3) are assigned to the quality model
quality_model.AssignParameters(results.loc[0,'params'])


# Evaluate the trained model on the training data and the validation data, to see how well the model predicts product quality from process measurements

y_train = []
y_val = []

# Estimation on training data cycles
for i in range(0,5): 
    c,y = quality_model.Simulation(data['init_state_train'][i], data['u_train'][i],None,data['switch_train'][i])
    plt.plot(np.array(c))
    # plt.plot(np.array(y))
    
    y = np.array(y[-1])[0,0]
    y_train.append(y)

#Estimation on validation data cycles    
for i in range(0,3): 
    _,y = quality_model.Simulation(data['init_state_val'][i], data['u_val'][i],None,data['switch_val'][i])
    y = np.array(y[-1])[0,0]
    y_val.append(y)


# Plot input validation data as well as predicted and true output data


fig, axs = plt.subplots(1,1)
fig.set_size_inches((13/2.54,8/2.54))
plt.plot([2,3,4,5,6],q_train,'d',markersize=12,label='train true')
plt.plot([7,8,9],q_val,'d',markersize=12,label='val true')
plt.plot([2,3,4,5,6],y_train,'d',markersize=12,label='train est')
plt.plot([7,8,9],y_val,'d',markersize=12,label='val est')
plt.xlabel('Zyklus')
plt.ylabel('Durchmesser_innen')

plt.tight_layout()
plt.legend()
plt.show()


# I.e. the model overestimates the true quality measurement "D innen" 
