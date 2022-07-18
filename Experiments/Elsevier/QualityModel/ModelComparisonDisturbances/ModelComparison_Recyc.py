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

GRU = pkl.load(open('GRU_c4_RecycDist_pred.pkl','rb'))
MLP = pkl.load(open('MLP_initstate_h2_RecycDist_pred.pkl','rb'))
Poly = pkl.load(open('Poly_p3_RecycDist_pred.pkl','rb'))

GRU = GRU.sort_index()
MLP = MLP.sort_index()
Poly = Poly.sort_index()

''' GET ERROR PERCENTILES (90% e.g.) FOR EACH MODEL ON VALIDATION DATA'''


# Un-normalize GRU predictions
mean_weight = 7.991
GRU['y_true'] = GRU['y_true']+mean_weight-1
GRU['y_est'] = GRU['y_est']+mean_weight-1

# Un-normalize MLP predictions
min_weight = 8.093
max_weight = 8.178

MLP['y_true'] = (MLP['y_true']+1) * (max_weight-min_weight) * 0.5 + min_weight 
MLP['y_est'] = (MLP['y_est']+1) * (max_weight-min_weight) * 0.5 + min_weight
MLP['e'] = MLP['y_true'] - MLP['y_est']




#Inititialize plot
color_map = sns.color_palette()
color_idx = 0
fig,ax = plt.subplots(1,1)

''' SORT ARRAY BEFORE PLOTTING ''' 


plt.plot(GRU.index, abs(GRU['e']),'k--o')
plt.plot(MLP.index, abs(MLP['e']),'b-x')
plt.plot(Poly.index, abs(Poly['e']),'r-.d')

# ax.set_xlabel('$n$')
# ax.set_ylabel('$\mathrm{BFR}$')

# plt.hlines(y=[], xmin=, xmax=10, colors='k', linestyles='dashed')

# xticks = [0,3] + list(np.arange(8,58+5,5)) + [62,67,72,76,81,86,90,95,100,
#                                               105,109,114,118]

# xticks = xticks[0::2]

# ax.set_xticks(ax.get_xticks()[xticks])

ax.set_xlim([-0.5,125])
ax.set_ylim([-0.001,0.051])

ax.set_xlabel('$c$')
ax.set_ylabel(None)


fig.set_size_inches((15/2.54,6/2.54))

plt.tight_layout()

plt.savefig('ModelComparison_Disturbance_Recyc.png', bbox_inches='tight',dpi=600)  