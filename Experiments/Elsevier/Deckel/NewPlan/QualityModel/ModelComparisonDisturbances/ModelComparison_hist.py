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



# Load predictions of the models that performed best on the disturbance case studies
GRU_switch = pkl.load(open('GRU_c4_SwitchDist_pred.pkl','rb'))
GRU_recyc = pkl.load(open('GRU_c4_RecycDist_pred.pkl','rb'))
GRU_val = pkl.load(open('GRU_c4_Val_pred.pkl','rb'))



MLP = pkl.load(open('MLP_initstate_h2_SwitchDist_pred.pkl','rb'))
Poly = pkl.load(open('Poly_p2_SwitchDist_pred.pkl','rb'))



# Un-normalize GRU predictions
mean_weight = 7.991
GRU_switch['y_true'] = GRU_switch['y_true']+mean_weight-1
GRU_switch['y_est'] = GRU_switch['y_est']+mean_weight-1

GRU_val['y_true'] = GRU_val['y_true']+mean_weight-1
GRU_val['y_est'] = GRU_val['y_est']+mean_weight-1

GRU_recyc['y_true'] = GRU_recyc['y_true']+mean_weight-1
GRU_recyc['y_est'] = GRU_recyc['y_est']+mean_weight-1

# # Un-normalize MLP predictions
# min_weight = 8.094
# max_weight = 8.178

# MLP['y_true'] = (MLP['y_true']+1) * (max_weight-min_weight) * 0.5 + min_weight 
# MLP['y_est'] = (MLP['y_est']+1) * (max_weight-min_weight) * 0.5 + min_weight
# MLP['e'] = MLP['y_true'] - MLP['y_est']

# MLP_e = [MLP_e_90,MLP_e_95,MLP_e_99]

# for i in range(len(MLP_e)):
#     MLP_e[i] = MLP_e[i]*(max_weight-min_weight)*0.5

# MLP_e_90 = MLP_e[0]
# MLP_e_95 = MLP_e[1]
# MLP_e_99 = MLP_e[2] 

#Inititialize plot

plt.close('all')

color_map = sns.color_palette()
fig,ax = plt.subplots(2,2)

sns.histplot(data=GRU_val,x='e',stat='proportion',color=color_map[0],
             bins=15,ax=ax[0,0])
sns.histplot(data=GRU_switch.loc[251:280],x='e',stat='proportion',color=color_map[1],
             bins=15,ax=ax[0,0])

sns.histplot(data=GRU_val,x='e',stat='proportion',color=color_map[0],
             bins=15,ax=ax[1,0])
sns.histplot(data=GRU_recyc.loc[1:20],x='e',stat='proportion',color=color_map[2],
             bins=15,ax=ax[1,0])


sns.scatterplot(x=GRU_val['y_true'],y=GRU_val['e'],ax=ax[0,1],
                color=color_map[0])
sns.scatterplot(x=GRU_switch['y_true'],y=GRU_switch['e'],ax=ax[0,1],
                color=color_map[1])

sns.scatterplot(x=GRU_val['y_true'],y=GRU_val['e'],ax=ax[1,1],
                color=color_map[0])
sns.scatterplot(x=GRU_recyc['y_true'],y=GRU_recyc['e'],ax=ax[1,1],
                color=color_map[2])


ax[0,0].set_xlim([-0.02,0.02])
ax[1,0].set_xlim([-0.02,0.02])

ax[0,1].set_xlim([8.09,8.18])
ax[1,1].set_xlim([8.09,8.18])

ax[0,1].set_ylim([-0.02,0.04])
ax[1,1].set_ylim([-0.02,0.04])

ax[0,0].legend(['Validierungsdaten','Nominalbetrieb 04.05.'])
ax[1,0].legend(['Validierungsdaten','Nominalbetrieb 06.05.'])

fig.set_size_inches((20/2.54,12/2.54))

plt.tight_layout()
plt.savefig('GRU_prediction_dist_val_hist.png', bbox_inches='tight',dpi=600)  

# ax[0].plot(GRU.index, abs(GRU['e']),color=color_map[4],linestyle='solid',marker='d')
# ax[0].hlines(y=[GRU_e_90,GRU_e_95,GRU_e_99], xmin=0, xmax=160, colors='grey', 
#              linestyles='dashed')


# ax[1].plot(MLP.index, abs(MLP['e']),color=color_map[3],linestyle='solid',marker='+')
# ax[1].hlines(y=[MLP_e_90,MLP_e_95,MLP_e_99], xmin=0, xmax=160, colors='grey', 
#              linestyles='dashed')

# ax[2].plot(Poly.index, abs(Poly['e']),color=color_map[0],linestyle='solid',
#            marker='o',markersize=5)
# ax[2].hlines(y=[Poly_e_90,Poly_e_95,Poly_e_99], xmin=0, xmax=160, colors='grey', 
#              linestyles='dashed')



# # Plot lines at time instances where disturbance was varied
# for a in ax:
#     a.vlines(x=[30,42,54,69,84,100,115], 
#            ymin=0, ymax=10, colors='k', linestyles='dashed')
#     a.set_xlim([-0.5,160])
#     a.set_ylim([-0.001,0.031])
#     a.set_ylabel('$\Delta m$' + ' in g')
#     a.set_xticks([1,20,40,60,80,100,120,140,160])
    
    
# ax[0].set_xticklabels([])
# ax[1].set_xticklabels([])
    
# ax[2].set_xlabel('$c$')

# fig.set_size_inches((15/2.54,10/2.54))

# plt.tight_layout()

# plt.savefig('ModelComparison_Disturbance_Switch.png', bbox_inches='tight',dpi=600)  


# for k in [10,20,30,40,50,60,70,80,90,100,110,120]:
#     print(BestFitRate(Poly.loc[1:k,'y_true'].values.reshape((-1,1)),
#                                 Poly.loc[1:k,'y_est'].values.reshape((-1,1))))

# ax.set_xlabel('$n$')
# ax.set_ylabel('$\mathrm{BFR}$')

# plt.hlines(y=[], xmin=, xmax=10, colors='k', linestyles='dashed')

# xticks = [0,3] + list(np.arange(8,58+5,5)) + [62,67,72,76,81,86,90,95,100,
#                                               105,109,114,118]

# xticks = xticks[0::2]

# ax.set_xticks(ax.get_xticks()[xticks])

#Inititialize plot
# fig,ax = plt.subplots(1,1)

# BFR_GRU = []
# BFR_MLP = []
# BFR_Poly = []

# for k in GRU.index[10::]:

#     BFR_GRU.append()























