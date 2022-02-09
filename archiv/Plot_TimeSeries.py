# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 10:00:28 2022

@author: LocalAdmin
"""

import pickle as pkl
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.rcParams['text.usetex'] = True

from DIM.miscellaneous.PreProcessing import arrange_data_for_ident

# Load Versuchsplan to find cycles that should be considered for modelling
c1 = pkl.load(open('./data/Versuchsplan/cycle1.pkl','rb'))




plt.close('all')
fig, ax1 = plt.subplots()
fig.set_size_inches((8.7/2.54,5.0/2.54))
ax1.set_xlim([0,15])
ax1.set_ylim([0,120])


ax1.set_xticklabels([])
ax1.set_yticklabels([])

ax1.set_xticks([])
ax1.set_yticks([])

ax1.fill_between(c1.index,c1['T_wkz_ist'], alpha=0.4,color=sns.color_palette()[1])
ax1.plot(c1['T_wkz_ist'],label = '$T_wkz_ist$',linewidth=2,
         color=sns.color_palette()[1])

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_yticklabels([])
ax2.set_yticks([])
ax2.set_ylim([0,650])


ax2.fill_between(c1.index,c1['p_wkz_ist'], alpha=0.8,color=sns.color_palette()[0])
ax2.plot(c1['p_wkz_ist'],label = '$p_wkz_ist$',linewidth=2,color=sns.color_palette()[0])

fig.tight_layout()
fig.savefig('p_wkz_T_wkz.png', bbox_inches='tight',dpi=600)