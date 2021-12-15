# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 16:45:55 2021

@author: alexa
"""
import pickle as pkl
import numpy as np

from DIM.models.model_structures import LSTM
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
    
    for charge in range(1,274):
        cycles_train_label.append(data[data['Charge']==charge].index.values[-6:-1])
        cycles_val_label.append(data[data['Charge']==charge].index.values[-1])
    
    cycles_train_label = np.hstack(cycles_train_label)
    cycles_val_label = np.hstack(cycles_val_label)
    
    
    # Delete cycles that for some reason don't exist
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
                                        u_inj_lab,u_press_lab,u_cool_lab,'quality')
    #
    # x_train,q_train,switch_train = arrange_data_for_qual_ident(cycles_train,x_lab,q_lab)
    
    x_val,q_val,switch_val = arrange_data_for_ident(cycles_val,y_lab,
                                        u_inj_lab,u_press_lab,u_cool_lab,'quality')
    
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
    
    return data,cycles_train_label,cycles_val_label

data,cycles_train_label,cycles_val_label = LoadData(dim_c=7)

path = 'E:/GitHub/DigitalTwinInjectionMolding/temp/PSO_param/q_model_Durchmesser_innen/'


# Load PSO results
hist = pkl.load(open(path+'HyperParamPSO_hist.pkl','rb'))
particle = pkl.load(open(path+'particle[7 3].pkl','rb'))
param_7_3 = particle.loc[10].params #hist.loc[7,3].model_params[0]                                       

# Initialize model structure
injection_model = LSTM(dim_u=5,dim_c=7,dim_hidden=5,dim_out=1,name='inject')
press_model = LSTM(dim_u=5,dim_c=7,dim_hidden=5,dim_out=1,name='press')
cool_model = LSTM(dim_u=2,dim_c=7,dim_hidden=5,dim_out=1,name='cool')

quality_model = QualityModel(subsystems=[injection_model,press_model,cool_model],
                              name='q_model_Durchmesser_innen')

quality_model.AssignParameters(param_7_3)


# y_train = []
y_val = []
e_val = []

#Estimation on training data cycles
# for i in range(0,1362): 
#     _,y = quality_model.Simulation(data['init_state_train'][i], data['u_train'][i],None,data['switch_train'][i])
#     y = np.array(y[-1])[0,0]
#     y_train.append(y)

#Estimation on validation data cycles    
for i in range(0,273): 
    _,y = quality_model.Simulation(data['init_state_val'][i], data['u_val'][i],None,data['switch_val'][i])
    y = np.array(y[-1])[0,0]
    y_val.append(y)
    e_val.append(data['y_val'][i]-y)

























