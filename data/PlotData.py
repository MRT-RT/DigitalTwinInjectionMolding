# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 09:19:36 2022

@author: alexa
"""

import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pathlib import Path

path = Path.cwd()
path_DIM = path.parents[0]

import sys
sys.path.insert(0, path_DIM.as_posix())

from DIM.miscellaneous.PreProcessing import LoadFeatureData

# data_path = path/'Versuchsplan/normalized/'

# plan = pkl.load(open( (data_path/'Versuchsplan.pkl').as_posix(),'rb' ))


data_train,data_test = LoadFeatureData((path/'Versuchsplan/').as_posix(),
[131,144], 'all',False)

data = pd.concat([data_train,data_test])

cycles_131 = list(data[data['Charge']==131].index)
cycles_144 = list(data[data['Charge']==144].index)

# load process data

process_data = {}

for cyc in cycles_131+cycles_144:
    
    cyc_path = path/'Versuchsplan/normalized/cycle'
    
    cyc_data = pkl.load(open(cyc_path.as_posix()+str(cyc)+'.pkl','rb'))

    
    process_data[cyc] = cyc_data


plt.close('all')

fig,ax = plt.subplots(2,2)

colormap131 = sns.color_palette("flare",n_colors=10)
colormap144 = sns.color_palette("crest",n_colors=10)


[ax[0,0].plot(i,data.loc[cycles_131[i],'Durchmesser_innen'],'d',
              color=colormap131[i]) for i in range(len(cycles_131)) ]
[ax[0,0].plot(i,data.loc[cycles_144[i],'Durchmesser_innen'],'d',
              color=colormap144[i]) for i in range(len(cycles_144)) ]


[ax[1,0].plot(data.loc[cycles_131[i],'T_wkz_0'],
              data.loc[cycles_131[i],'Durchmesser_innen'],'d',
              color=colormap131[i]) for i in range(len(cycles_131)) ]
[ax[1,0].plot(data.loc[cycles_144[i],'T_wkz_0'],
              data.loc[cycles_144[i],'Durchmesser_innen'],'d',
              color=colormap144[i]) for i in range(len(cycles_144)) ]


# p_machine = ['V_Screw_ist']
p_machine = ['Q_Vol_ist','V_Screw_ist','p_inj_ist']
p_cav = ['p_wkz_ist', 'T_wkz_ist']

[ax[0,1].plot(process_data[cycles_131[i]][p_machine],color = colormap131[i]) 
for i in range(len(cycles_131)) ]

[ax[1,1].plot(process_data[cycles_131[i]][p_cav],color = colormap131[i]) 
for i in range(len(cycles_131)) ]


[ax[0,1].plot(process_data[cycles_144[i]][p_machine],color = colormap144[i]) 
for i in range(len(cycles_144)) ]

[ax[1,1].plot(process_data[cycles_144[i]][p_cav],color = colormap144[i]) 
for i in range(len(cycles_144)) ]

fig,ax = plt.subplots(2,2)

colormap131 = sns.color_palette("flare",n_colors=10)
colormap144 = sns.color_palette("crest",n_colors=10)


[ax[0,0].plot(i,data.loc[cycles_131[i],'Gewicht'],'d',
              color=colormap131[i]) for i in range(len(cycles_131)) ]
[ax[0,0].plot(i,data.loc[cycles_144[i],'Gewicht'],'d',
              color=colormap144[i]) for i in range(len(cycles_144)) ]


[ax[1,0].plot(data.loc[cycles_131[i],'T_wkz_0'],
              data.loc[cycles_131[i],'Gewicht'],'d',
              color=colormap131[i]) for i in range(len(cycles_131)) ]
[ax[1,0].plot(data.loc[cycles_144[i],'T_wkz_0'],
              data.loc[cycles_144[i],'Gewicht'],'d',
              color=colormap144[i]) for i in range(len(cycles_144)) ]


# p_machine = ['V_Screw_ist']
p_machine = ['Q_Vol_ist','V_Screw_ist','p_inj_ist']
p_cav = ['p_wkz_ist', 'T_wkz_ist']

[ax[0,1].plot(process_data[cycles_131[i]][p_machine],color = colormap131[i]) 
for i in range(len(cycles_131)) ]

[ax[1,1].plot(process_data[cycles_131[i]][p_cav],color = colormap131[i]) 
for i in range(len(cycles_131)) ]


[ax[0,1].plot(process_data[cycles_144[i]][p_machine],color = colormap144[i]) 
for i in range(len(cycles_144)) ]

[ax[1,1].plot(process_data[cycles_144[i]][p_cav],color = colormap144[i]) 
for i in range(len(cycles_144)) ]