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


path_dim = Path.cwd().parents[1]
sys.path.insert(0, path_dim.as_posix())

from DIM.miscellaneous.PreProcessing import LoadFeatureData, MinMaxScale

# from DIM.models.model_structures import GRU
# from DIM.models.injection_molding import QualityModel
from DIM.optim.common import BestFitRate

from functions import Eval_MLP


MLP_1layer_T0 = pkl.load(open('MLP_1layer_T0/Pred_MLP_1layer_T0_h10_init19.pkl','rb'))
MLP_2layers_T0 = pkl.load(open('MLP_2layers_T0/Pred_MLP_2layers_T0_h10_init4.pkl','rb'))

MLP_1layer_p0 = pkl.load(open('MLP_1layer_p0/Pred_MLP_1layer_p0_h10_init9.pkl','rb'))

MLP_2layers_p0 = pkl.load(open('MLP_2layers_p0/Pred_MLP_2layers_p0_h10_init13.pkl','rb'))

MLP_2layers_static = pkl.load(open('MLP_2layers_static/Pred_MLP_2layers_static_h10_init7.pkl','rb'))
# %%

Di = ['Durchmesser_innen']
Di_est = ['Durchmesser_innen_est']


# %% Plot predictions

plt.close('all')
color_map = sns.color_palette()
fig,ax = plt.subplots(5,1)

kwargs_true = {'linestyle':'None','marker':'d','markersize':4}
kwargs_est = {'linestyle':'None','marker':'o','markersize':2}

i = 0

for M in [MLP_1layer_T0,MLP_2layers_T0,MLP_1layer_p0,MLP_2layers_p0,
          MLP_2layers_static]:

    ax[i].plot(MLP_1layer_T0['results_train']['pred'].index,
               MLP_1layer_T0['results_train']['pred'][Di].values,
               color='grey',**kwargs_true)
    ax[i].plot(MLP_1layer_T0['results_train']['pred'].index,
               MLP_1layer_T0['results_train']['pred'][Di_est].values,
               color = color_map[i],**kwargs_est)
    
    ax[i].plot(MLP_1layer_T0['results_val']['pred'].index,
               MLP_1layer_T0['results_val']['pred'][Di].values,
               color='red',**kwargs_true)
    ax[i].plot(MLP_1layer_T0['results_val']['pred'].index,
               MLP_1layer_T0['results_val']['pred'][Di_est].values,
               color = color_map[i],**kwargs_est)

    i = i + 1


# %%
for a in ax:
    a.set_xlim([0,100])#([1800,1900])
    a.set_ylim([27.2,28])
    a.set_xlabel('$c$')
    a.set_ylabel('$D_{\mathrm{i}}$ in $\mathrm{mm}$')


# %%
ax[0].set_xticklabels([])
ax[1].set_xticklabels([])
ax[2].set_xticklabels([])
ax[3].set_xlabel('$c$')

# %%

ax[0].legend(['$\mathcal{D}_{\mathrm{train}}$','$\mathcal{D}_{\mathrm{val}}$',
'$\mathrm{ID}^{4}_{3}$'])
ax[1].legend(['$\mathcal{D}_{\mathrm{train}}$','$\mathcal{D}_{\mathrm{val}}$',
'$\mathrm{MLP}^{10}_{\mathrm{s}}$'])
ax[2].legend(['$\mathcal{D}_{\mathrm{train}}$','$\mathcal{D}_{\mathrm{val}}$',
'$\mathrm{PR}^{4}_{\mathrm{s}}$' + ' with ' + '$x_{0}$'])

fig.set_size_inches((15/2.54,12/2.54))

plt.tight_layout()

# plt.savefig('Di_MLP_predict.png', bbox_inches='tight',dpi=600)  

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