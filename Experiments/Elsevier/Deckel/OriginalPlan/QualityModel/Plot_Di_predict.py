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
from pathlib import Path

path_dim = Path.cwd().parents[4]
sys.path.insert(0, path_dim.as_posix())

from DIM.miscellaneous.PreProcessing import LoadFeatureData, MinMaxScale

# from DIM.models.model_structures import GRU
# from DIM.models.injection_molding import QualityModel
from DIM.optim.common import BestFitRate
# from DIM.optim.param_optim import parallel_mode
# from DIM.miscellaneous.PreProcessing import arrange_data_for_ident, eliminate_outliers, LoadDynamicData

# Load data used for normalization

path = path_dim / 'data/Versuchsplan/normalized/'

plan = pkl.load(open(path.as_posix() + '/Versuchsplan.pkl','rb'))

data_train,data_val = LoadFeatureData(path.as_posix(),list(range(1,275)),

                                      'all',True)
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

ax[0].plot(GRU[0].index, GRU[0]['y_true'],color='grey',linestyle='None',marker='d')
ax[0].plot(GRU[1].index, GRU[1]['y_true'],color='red',linestyle='None',marker='d')
ax[0].plot(GRU[0].index, GRU[0]['y_est'],color=color_map[4],linestyle='None',marker='o')
ax[0].plot(GRU[1].index, GRU[1]['y_est'],color=color_map[4],linestyle='None',marker='o')


ax[1].plot(MLP[0].index, MLP[0]['y_true'],color='grey',linestyle='None',marker='d')
ax[1].plot(MLP[1].index, MLP[1]['y_true'],color='red',linestyle='None',marker='d')
ax[1].plot(MLP[0].index, MLP[0]['y_est'],color=color_map[2],linestyle='None',marker='o')
ax[1].plot(MLP[1].index, MLP[1]['y_est'],color=color_map[2],linestyle='None',marker='o')

ax[2].plot(Poly[0].index, Poly[0]['y_true'],color='grey',linestyle='None',marker='d')
ax[2].plot(Poly[1].index, Poly[1]['y_true'],color='red',linestyle='None',marker='d')
ax[2].plot(Poly[0].index, Poly[0]['y_est'],color=color_map[1],linestyle='None',marker='o')
ax[2].plot(Poly[1].index, Poly[1]['y_est'],color=color_map[1],linestyle='None',marker='o')


for a in ax:
    a.set_xlim([0,100])#([1800,1900])
    a.set_ylim([27.2,28])
    a.set_xlabel('$c$')
    a.set_ylabel('$D_{\mathrm{i}}$ in $\mathrm{mm}$')


ax[0].set_xticklabels([])
ax[1].set_xticklabels([])
ax[2].set_xlabel('$c$')


ax[0].legend(['$\mathcal{D}_{\mathrm{train}}$','$\mathcal{D}_{\mathrm{val}}$',
'$\mathrm{ID}^{4}_{3}$'])
ax[1].legend(['$\mathcal{D}_{\mathrm{train}}$','$\mathcal{D}_{\mathrm{val}}$',
'$\mathrm{MLP}^{10}_{\mathrm{s}}$'])
ax[2].legend(['$\mathcal{D}_{\mathrm{train}}$','$\mathcal{D}_{\mathrm{val}}$',
'$\mathrm{PR}^{4}_{\mathrm{s}}$' + ' with ' + '$x_{0}$'])





# legend = stripplot.get_legend()
# legend.set_title(None)

# legend_texts = ['$\mathrm{PR}^{n}_{\mathrm{s}}$',
#             '$\mathrm{PR}^{n}_{\mathrm{s}}$' + ' with ' + '$x_{0}$',
#             '$\mathrm{MLP}^{n}_{\mathrm{s}}$',
#             '$\mathrm{MLP}^{n}_{\mathrm{s}}$' + ' with '  + '$x_{0}$',
#             '$\mathrm{ID}^{n}_{3}$']


# # ax.set_xticks(ax.get_xticks()[xticks])
# # ax.set_xlim([-0.5,125])
# # ax.set_ylim([-0.001,0.051])
# ax.set_ylabel('$D_{\mathrm{i}}$' + ' in ' + '$\mathrm{mm}$' )
    
    
# ax.set_xlabel('$c$')

fig.set_size_inches((15/2.54,12/2.54))

plt.tight_layout()

# plt.savefig('Di_predict.png', bbox_inches='tight',dpi=600)  



fig2,ax2 = plt.subplots(1,3)

kwargs = {'binwidth': 0.01, 'stat': 'probability'}

sns.histplot(GRU[1]['e'],color=color_map[4],ax=ax2[0],**kwargs)
sns.histplot(MLP[1]['e'],color=color_map[2],ax=ax2[1],**kwargs)
sns.histplot(Poly[1]['e'],color=color_map[1],ax=ax2[2],**kwargs)

[a.set_xlim([-0.3,0.3]) for a in ax2]
[a.set_ylim([0,0.13]) for a in ax2]
# [a.set_yticklabels([0,0.13]) for a in ax2]
[a.set_xlabel('$e$') for a in ax2]
[a.set_ylabel(None) for a in ax2]


legends = [['$\mathrm{ID}^{4}_{3}$'],['$\mathrm{MLP}^{10}_{\mathrm{s}}$'],
          ['$\mathrm{PR}^{4}_{\mathrm{s}}$' + ' with ' + '$x_{0}$']]

[a.legend(l) for a,l in zip(ax2,legends)]
ax[0].legend(['$\mathcal{D}_{\mathrm{train}}$','$\mathcal{D}_{\mathrm{val}}$',
'$\mathrm{ID}^{4}_{3}$'])
ax[1].legend(['$\mathcal{D}_{\mathrm{train}}$','$\mathcal{D}_{\mathrm{val}}$',
])
ax[2].legend(['$\mathcal{D}_{\mathrm{train}}$','$\mathcal{D}_{\mathrm{val}}$',
'$\mathrm{PR}^{4}_{\mathrm{s}}$' + ' with ' + '$x_{0}$'])



fig.set_size_inches((15/2.54,6/2.54))

plt.tight_layout()
# plt.savefig('Di_predict_hist.png', bbox_inches='tight',dpi=600)




