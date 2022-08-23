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


# Un-normalize GRU
GRU['y_true'] = GRU['y_true']+7.991-1
GRU['y_est'] = GRU['y_est']+7.991-1


#Inititialize plot
color_map = sns.color_palette()
color_idx = 0
fig,ax = plt.subplots(1,1)

for i in range(4,5):

    results_train,results_st = Eval_GRU_on_Val(dim_c=i)
    
    print(BestFitRate(results_st['y_true'].values.reshape((-1,1)),
                results_st['y_est'].values.reshape((-1,1))))
    
    e = abs(results_st['y_true']-results_st['y_est'])
    
    sns.stripplot(x = results_st.index, 
                  y= e,
                  ax=ax,color=color_map[color_idx],label = str(i))
    
    print(np.percentile(e, 0.9))
    
    # color_idx = color_idx +1
    
    pkl.dump(results_st,open('GRU_c'+str(i)+'_RecycDist_pred.pkl','wb'))
    
    
fig,ax = plt.subplots(1,1)

weight_c11 = 7.991

sns.stripplot(x = results_st.index, y=results_st['y_true']+weight_c11-1,
              ax=ax,color='grey')

# ax.set_xlabel('$n$')
# ax.set_ylabel('$\mathrm{BFR}$')

plt.vlines(x=[results_st.index.get_loc(loc) for loc in [20,45,65,85,105]], 
           ymin=0, ymax=10, colors='k', linestyles='dashed')

xticks = [0,3] + list(np.arange(8,58+5,5)) + [62,67,72,76,81,86,90,95,100,
                                              105,109,114,118]

xticks = xticks[0::2]

ax.set_xticks(ax.get_xticks()[xticks])

ax.set_xlim([-0.5,125])
ax.set_ylim([8.1,8.2])

ax.set_xlabel('$c$')
ax.set_ylabel(None)


fig.set_size_inches((15/2.54,6/2.54))

plt.tight_layout()

plt.savefig('Data_Disturbance_Recyc.png', bbox_inches='tight',dpi=600)    
    # plt.figure()
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