# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 16:45:55 2021

@author: alexa
"""
# %% Load libraries

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

# %% Load data 

path_1 = path_dim / 'data/Stoergroessen/20220504/Versuchsplan'
path_2 = path_dim / 'data/Stoergroessen/20220504/Umschaltpkt_Stoerung'
path_3 = path_dim / 'data/Stoergroessen/20220506/Rezyklat_Stoerung'

# plan = pkl.load(open(path.as_posix() + '/Versuchsplan.pkl','rb'))

# data_train,data_val = LoadFeatureData(path.as_posix(),list(range(1,275)),
#                                       'all',True)

data_plan = pd.concat(LoadFeatureData(path_1.as_posix(),list(range(1,26)),
                                      'all',True))
# 
# data_all = pd.concat([data_train,data_val])

# _,minmax = MinMaxScale(data_train,['Durchmesser_innen'])


plt.close('all')


# # Load Predictions of models
# GRU = pkl.load(open('./GRU/Durchmesser_innen/GRU_4c_pred.pkl','rb'))
# MLP = pkl.load(open('./setpoint_models/Durchmesser_innen/MLP_two_layers/MLP_2l_h10_pred.pkl','rb'))
# Poly = pkl.load(open('./setpoint_models/Durchmesser_innen/setpoints_initial_state/Poly_p4_pred.pkl','rb'))

# # GRU = GRU.sort_index()
# # MLP = MLP.sort_index()
# # Poly = Poly.sort_index()

# # # Un-normalize GRU predictions
# mean_weight = plan.loc[11,'Durchmesser_innen']
# for df in GRU:
#     df['y_true'] = df['y_true']+mean_weight-1
#     df['y_est'] = df['y_est']+mean_weight-1

# # Un-normalize MLP predictions
# min_weight = minmax[0]['Durchmesser_innen']
# max_weight = minmax[1]['Durchmesser_innen']

# for df in MLP:
#     df['y_true'] = (df['y_true']+1) * (max_weight-min_weight) * 0.5 + min_weight 
#     df['y_est'] = (df['y_est']+1) * (max_weight-min_weight) * 0.5 + min_weight
#     df['e'] = df['y_true'] - df['y_est']



# %% Plot predictions
color_map = sns.color_palette()
fig,ax = plt.subplots(3,1)


kwargs_true = {'linestyle':'None','marker':'d','markersize':4}
kwargs_est = {'linestyle':'None','marker':'o','markersize':2}

ax[0].plot(GRU[0].index, GRU[0]['y_true'],color='grey',**kwargs_true)
ax[0].plot(GRU[1].index, GRU[1]['y_true'],color='red',**kwargs_true)
ax[0].plot(GRU[0].index, GRU[0]['y_est'],color=color_map[4],**kwargs_est)
ax[0].plot(GRU[1].index, GRU[1]['y_est'],color=color_map[4],**kwargs_est)


ax[1].plot(MLP[0].index, MLP[0]['y_true'],color='grey',**kwargs_true)
ax[1].plot(MLP[1].index, MLP[1]['y_true'],color='red',**kwargs_true)
ax[1].plot(MLP[0].index, MLP[0]['y_est'],color=color_map[2],**kwargs_est)
ax[1].plot(MLP[1].index, MLP[1]['y_est'],color=color_map[2],**kwargs_est)

ax[2].plot(Poly[0].index, Poly[0]['y_true'],color='grey',**kwargs_true)
ax[2].plot(Poly[1].index, Poly[1]['y_true'],color='red',**kwargs_true)
ax[2].plot(Poly[0].index, Poly[0]['y_est'],color=color_map[1],**kwargs_est)
ax[2].plot(Poly[1].index, Poly[1]['y_est'],color=color_map[1],**kwargs_est)


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

fig.set_size_inches((15/2.54,12/2.54))

plt.tight_layout()

plt.savefig('Di_predict.png', bbox_inches='tight',dpi=600)  

# %% Plot error histogram
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
plt.savefig('Di_predict_hist.png', bbox_inches='tight',dpi=600)

# %% Plot scatter

fig3,ax3 = plt.subplots(1,3)
sns.scatterplot(data=GRU[1],x='y_est',y='e',color=color_map[4],ax=ax3[0])
sns.scatterplot(data=MLP[1],x='y_est',y='e',color=color_map[2],ax=ax3[1])
sns.scatterplot(data=Poly[1],x='y_est',y='e',color=color_map[1],ax=ax3[2])

[a.set_xlim([27.25,28]) for a in ax3]
[a.set_ylim([-0.24,0.28]) for a in ax3]
# [a.set_yticklabels([0,0.13]) for a in ax2]
[a.set_xlabel('$\hat{D}_{i}$') for a in ax3]

legends = [['$\mathrm{ID}^{4}_{3}$'],['$\mathrm{MLP}^{10}_{\mathrm{s}}$'],
          ['$\mathrm{PR}^{4}_{\mathrm{s}}$' + ' with ' + '$x_{0}$']]

[a.legend(l) for a,l in zip(ax3,legends)]
ax[0].legend(['$\mathcal{D}_{\mathrm{train}}$','$\mathcal{D}_{\mathrm{val}}$',
'$\mathrm{ID}^{4}_{3}$'])
ax[1].legend(['$\mathcal{D}_{\mathrm{train}}$','$\mathcal{D}_{\mathrm{val}}$',
])
ax[2].legend(['$\mathcal{D}_{\mathrm{train}}$','$\mathcal{D}_{\mathrm{val}}$',
'$\mathrm{PR}^{4}_{\mathrm{s}}$' + ' with ' + '$x_{0}$'])

fig.set_size_inches((15/2.54,6/2.54))

plt.tight_layout()
plt.savefig('Di_est_e_scatter.png', bbox_inches='tight',dpi=600)

# %% Plot scatter

fig4,ax4 = plt.subplots(1,3)
sns.scatterplot(data=GRU[1],x='y_true',y='e',color=color_map[4],ax=ax4[0])
sns.scatterplot(data=MLP[1],x='y_true',y='e',color=color_map[2],ax=ax4[1])
sns.scatterplot(data=Poly[1],x='y_true',y='e',color=color_map[1],ax=ax4[2])

[a.set_xlim([27.25,28]) for a in ax4]
[a.set_ylim([-0.24,0.28]) for a in ax4]
# [a.set_yticklabels([0,0.13]) for a in ax2]
[a.set_xlabel('$D_{i}$') for a in ax4]


legends = [['$\mathrm{ID}^{4}_{3}$'],['$\mathrm{MLP}^{10}_{\mathrm{s}}$'],
          ['$\mathrm{PR}^{4}_{\mathrm{s}}$' + ' with ' + '$x_{0}$']]

[a.legend(l) for a,l in zip(ax4,legends)]
ax[0].legend(['$\mathcal{D}_{\mathrm{train}}$','$\mathcal{D}_{\mathrm{val}}$',
'$\mathrm{ID}^{4}_{3}$'])
ax[1].legend(['$\mathcal{D}_{\mathrm{train}}$','$\mathcal{D}_{\mathrm{val}}$',
])
ax[2].legend(['$\mathcal{D}_{\mathrm{train}}$','$\mathcal{D}_{\mathrm{val}}$',
'$\mathrm{PR}^{4}_{\mathrm{s}}$' + ' with ' + '$x_{0}$'])

fig.set_size_inches((15/2.54,6/2.54))

plt.tight_layout()
plt.savefig('Di_true_e_scatter.png', bbox_inches='tight',dpi=600)


# %% e vs T0

fig5,ax5 = plt.subplots(1,3)

GRU_plot = pd.concat(GRU)
GRU_plot = pd.concat([GRU_plot,data_all['T_wkz_0']],axis=1)

MLP_plot = pd.concat(MLP)
MLP_plot = pd.concat([MLP_plot,data_all['T_wkz_0']],axis=1)

Poly_plot = pd.concat(Poly)
Poly_plot = pd.concat([Poly_plot,data_all['T_wkz_0']],axis=1)

sns.scatterplot(data=GRU_plot,x='T_wkz_0',y='e',color=color_map[4],ax=ax5[0])
sns.scatterplot(data=MLP_plot,x='T_wkz_0',y='e',color=color_map[2],ax=ax5[1])
sns.scatterplot(data=Poly_plot,x='T_wkz_0',y='e',color=color_map[1],ax=ax5[2])

# [a.set_xlim([-0.2,0.4]) for a in ax5]
[a.set_ylim([-0.3,0.3]) for a in ax5]
# [a.set_yticklabels([0,0.13]) for a in ax2]
# [a.set_xlabel('$D_{i}$') for a in ax5]


legends = [['$\mathrm{ID}^{4}_{3}$'],['$\mathrm{MLP}^{10}_{\mathrm{s}}$'],
          ['$\mathrm{PR}^{4}_{\mathrm{s}}$' + ' with ' + '$x_{0}$']]

[a.legend(l) for a,l in zip(ax5,legends)]
ax[0].legend(['$\mathcal{D}_{\mathrm{train}}$','$\mathcal{D}_{\mathrm{val}}$',
'$\mathrm{ID}^{4}_{3}$'])
ax[1].legend(['$\mathcal{D}_{\mathrm{train}}$','$\mathcal{D}_{\mathrm{val}}$',
])
ax[2].legend(['$\mathcal{D}_{\mathrm{train}}$','$\mathcal{D}_{\mathrm{val}}$',
'$\mathrm{PR}^{4}_{\mathrm{s}}$' + ' with ' + '$x_{0}$'])

fig.set_size_inches((15/2.54,6/2.54))

plt.tight_layout()
plt.savefig('T0_e_scatter.png', bbox_inches='tight',dpi=600)


# %% Di over T0 GRU

num_plots = 3

fig6,ax6 = plt.subplots(num_plots,num_plots)

GRU_plot = pd.concat(GRU)
GRU_plot = pd.concat([GRU_plot,data_all[['T_wkz_0','Charge']]],axis=1)

kwargs_true = {'linestyle':'None','marker':'d','markersize':6}
kwargs_est = {'linestyle':'None','marker':'o','markersize':4}

x_min = GRU_plot['T_wkz_0'].min()
x_max = GRU_plot['T_wkz_0'].max()

for ch in range(1,num_plots**2+1):
    data_ch = GRU_plot.loc[GRU_plot['Charge']==ch]
    
    ax6.flat[ch-1].plot(data_ch['T_wkz_0'].values,data_ch['y_true'],color='grey',
                      **kwargs_true)
    ax6.flat[ch-1].plot(data_ch['T_wkz_0'].values,data_ch['y_est'],
                      color=color_map[4],**kwargs_est)
    
    ax6.flat[ch-1].set_xlim((50,65))
    ax6.flat[ch-1].set_yticks([27.2,27.9])
    ax6.flat[ch-1].set_ylim((27.2,27.9))
# sns.scatterplot(data=GRU_plot,x='T_wkz_0',y='e',color=color_map[4],ax=ax5[0])
# sns.scatterplot(data=MLP_plot,x='T_wkz_0',y='e',color=color_map[2],ax=ax5[1])
# sns.scatterplot(data=Poly_plot,x='T_wkz_0',y='e',color=color_map[1],ax=ax5[2])

# # [a.set_xlim([-0.2,0.4]) for a in ax5]
# [a.set_ylim([-0.3,0.3]) for a in ax5]
# # [a.set_yticklabels([0,0.13]) for a in ax2]
# # [a.set_xlabel('$D_{i}$') for a in ax5]


# legends = [['$\mathrm{ID}^{4}_{3}$'],['$\mathrm{MLP}^{10}_{\mathrm{s}}$'],
#           ['$\mathrm{PR}^{4}_{\mathrm{s}}$' + ' with ' + '$x_{0}$']]

# [a.legend(l) for a,l in zip(ax5,legends)]
# ax[0].legend(['$\mathcal{D}_{\mathrm{train}}$','$\mathcal{D}_{\mathrm{val}}$',
# '$\mathrm{ID}^{4}_{3}$'])
# ax[1].legend(['$\mathcal{D}_{\mathrm{train}}$','$\mathcal{D}_{\mathrm{val}}$',
# ])
# ax[2].legend(['$\mathcal{D}_{\mathrm{train}}$','$\mathcal{D}_{\mathrm{val}}$',
# '$\mathrm{PR}^{4}_{\mathrm{s}}$' + ' with ' + '$x_{0}$'])

fig.set_size_inches((15/2.54,6/2.54))

plt.tight_layout()
# plt.savefig('T0_e_scatter.png', bbox_inches='tight',dpi=600)

# %% Plot cycle 131 144 (same setpoints)
fig7,ax7 = plt.subplots(2,1)

kwargs_true = {'linestyle':'None','marker':'d','markersize':6}

charge_131 = data_all.loc[data_all['Charge']==131]
charge_144 = data_all.loc[data_all['Charge']==144]

ax7[0].plot(charge_131['T_wkz_0'].values,charge_131['Durchmesser_innen'],
                    color='grey', **kwargs_true)

ax7[0].plot(charge_144['T_wkz_0'].values,charge_144['Durchmesser_innen'],
                    color='blue', **kwargs_true)


ax7[1].plot(charge_131['T_wkz_0'].values,charge_131['Gewicht'],
                    color='grey', **kwargs_true)

ax7[1].plot(charge_144['T_wkz_0'].values,charge_144['Gewicht'],
                    color='blue', **kwargs_true)

ax7[0].set_ylabel('$D_{\mathrm{i}}$')
ax7[1].set_ylabel('$m$')
ax7[1].set_xlabel('${\circ}C$')

fig7.set_size_inches((15/2.54,12/2.54))
plt.tight_layout()
plt.savefig('Di_T0_plot.png', bbox_inches='tight',dpi=600)