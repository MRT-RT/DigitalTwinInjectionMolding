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

from DIM.miscellaneous.PreProcessing import LoadFeatureData, MinMaxScale

# from DIM.models.model_structures import GRU
# from DIM.models.injection_molding import QualityModel
# from DIM.optim.common import BestFitRate
# from DIM.optim.param_optim import parallel_mode
# from DIM.miscellaneous.PreProcessing import arrange_data_for_ident, eliminate_outliers, LoadDynamicData

# Load data used for normalization

# path_sys = 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/'
# path_sys = 'C:/Users/LocalAdmin/Documents/GitHub/DigitalTwinInjectionMolding/'
path_sys = '/home/alexander/GitHub/DigitalTwinInjectionMolding/' 
# path_sys = 'E:/GitHub/DigitalTwinInjectionMolding/'

path = path_sys + '/data/Versuchsplan/normalized/'

plan = pkl.load(open(path+'Versuchsplan.pkl','rb'))

data_train,data_val = LoadFeatureData(path,list(range(1,275)),'all',True)
_,minmax = MinMaxScale(data_train,['Durchmesser_innen'])


plt.close('all')


# Load Predictions of models
GRU = pkl.load(open('./GRU/Durchmesser_innen/GRU_4c_pred.pkl','rb'))
MLP = pkl.load(open('./setpoint_models/Durchmesser_innen/MLP_two_layers/MLP_2l_h10_pred.pkl','rb'))
Poly = pkl.load(open('./setpoint_models/Durchmesser_innen/setpoints_initial_state/Poly_p4_pred.pkl','rb'))

# GRU = GRU.sort_index()
# MLP = MLP.sort_index()
# Poly = Poly.sort_index()

# # Un-normalize GRU predictions
mean_weight = plan.loc[11,'Durchmesser_innen']
for df in GRU:
    df['y_true'] = df['y_true']+mean_weight-1
    df['y_est'] = df['y_est']+mean_weight-1

# Un-normalize MLP predictions
min_weight = minmax[0]['Durchmesser_innen']
max_weight = minmax[1]['Durchmesser_innen']

for df in MLP:
    df['y_true'] = (df['y_true']+1) * (max_weight-min_weight) * 0.5 + min_weight 
    df['y_est'] = (df['y_est']+1) * (max_weight-min_weight) * 0.5 + min_weight
    df['e'] = df['y_true'] - df['y_est']

color_map = sns.color_palette()
fig,ax = plt.subplots(3,1)

ax[0].plot(GRU[0].index, GRU[0]['y_true'],color=color_map[0],linestyle='solid',marker='d')
ax[0].plot(MLP[0].index, MLP[0]['y_true'],color=color_map[1],linestyle='solid',marker='d')
ax[0].plot(Poly[0].index, Poly[0]['y_true'],color=color_map[2],linestyle='solid',marker='d')

# ax[0].hlines(y=[GRU_e_90,GRU_e_95,GRU_e_99], xmin=0, xmax=130, colors='grey', 
#              linestyles='dashed')












































# path_sys = 'C:/Users/LocalAdmin/Documents/GitHub/DigitalTwinInjectionMolding/'
# path_sys = '/home/alexander/GitHub/DigitalTwinInjectionMolding/'

# path = path_sys + 'data/Versuchsplan/'


# plan = pkl.load(open(path+'Versuchsplan.pkl','rb'))

# plan = plan.sort_index()

# #Inititialize plot
# color_map = sns.color_palette()
# fig,ax = plt.subplots(1,1)

# # sns.stripplot(x=plan.index[0:100], y=plan.loc[1:100,'Durchmesser_innen'],color='grey',ax=ax)

# ax.plot(plan.loc[1:101].index,plan.loc[1:101,'Durchmesser_innen'],
#         linestyle = 'None',marker='o',color='grey')

# x_tick_val = list(np.arange(2,102,10)) + list(np.arange(8,102,10))
# x_tick_val.sort()

# # Charge 8 has 11 reptitions, indices of following charges must be increased by 1
# for i in range(-4,-1):
#     x_tick_val[i] =  x_tick_val[i] + 1

# x_tick_val[-1] =  x_tick_val[-1] + 1

# # sns.stripplot(x=x_tick_val,y=plan.loc[x_tick_val,'Durchmesser_innen'],color=color_map[0],ax=ax)
# ax.plot(plan.loc[x_tick_val].index,plan.loc[x_tick_val,'Durchmesser_innen'],
#         linestyle = 'None',marker='o',color='red')


# # xticks = [0] + list(np.arange(9,100,10))

# # ax.set_xticks(ax.get_xticks()[xticks])
# # ax.set_xlim([-0.5,125])
# # ax.set_ylim([-0.001,0.051])
# ax.set_ylabel('$D_{\mathrm{i}}$' + ' in ' + '$\mathrm{mm}$' )
    
    
# ax.set_xlabel('$c$')

# fig.set_size_inches((15/2.54,6/2.54))

# plt.tight_layout()

# plt.savefig('Di_data.png', bbox_inches='tight',dpi=600)  








