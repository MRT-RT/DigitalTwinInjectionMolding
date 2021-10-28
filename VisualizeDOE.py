# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:46:17 2021

@author: alexa
"""

import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt

data = pkl.load(open('data/Versuchsplan/Versuchsplan.pkl','rb'))

# eliminate outliers
data_clean = data[data.loc[:,'Gewicht']>=5]
data_clean = data_clean[data_clean.loc[:,'Stegbreite_Gelenk']>=4]
data_clean = data_clean[data_clean.loc[:,'Breite_Lasche']>=4]

data = data_clean

# quality_meas = 'Gewicht'
quality_meas = 'Durchmesser_innen'
# quality_meas = 'Durchmesser_außen'
# quality_meas = 'Stegbreite_Gelenk'
# quality_meas = 'Breite_Lasche'
# quality_meas = 'Rundheit_außen'


num_bins = 30
# binwidth = 0.005 # Gewicht
binwidth = 0.01 # Durchmesser_innen

# Einfluss Einspritzgeschwindigkeit ###########################################
plt.close('all')
palette = sns.color_palette()

fig, axs = plt.subplots() #plt.subplots(2,gridspec_kw={'height_ratios': [1, 1.5]})
fig.set_size_inches((9/2.54,4/2.54))

sns.histplot(data=data, x=quality_meas,stat='probability',
             color=palette[0],bins=num_bins,
             binwidth=binwidth) # Histogramm of all data

data_subset = data.loc[data['Düsentemperatur'].isin([250]) & 
               data['Werkzeugtemperatur'].isin([40]) &
               data['Umschaltpunkt'].isin([14]) &
               data['Nachdruckhöhe'].isin([600]) &
               data['Nachdruckzeit'].isin([3]) &
               data['Staudruck'].isin([25]) &
               data['Kühlzeit'].isin([15])]

v16 = data_subset.loc[data_subset['Einspritzgeschwindigkeit'].isin([16])]
v48 = data_subset.loc[data_subset['Einspritzgeschwindigkeit'].isin([48])]

sns.histplot(data=v16, x=quality_meas,
             color=palette[1],stat='probability',bins=num_bins,
             binwidth=binwidth)
sns.histplot(data=v48, x=quality_meas,
             color=palette[2],stat='probability',bins=num_bins,
             binwidth=binwidth)

axs.set_xlabel(quality_meas)
axs.set_ylabel(None)
fig.tight_layout(pad=0.0, h_pad=None, w_pad=None, rect=None)

# Einfluss Düsentemperatur ####################################################
palette = sns.color_palette()

fig, axs = plt.subplots() #plt.subplots(2,gridspec_kw={'height_ratios': [1, 1.5]})
fig.set_size_inches((9/2.54,4/2.54))

sns.histplot(data=data, x=quality_meas,stat='probability',
             color=palette[0],bins=num_bins,
             binwidth=binwidth) # Histogramm of all data

data_subset = data.loc[data['Werkzeugtemperatur'].isin([40]) &
               data['Umschaltpunkt'].isin([14]) &
               data['Nachdruckhöhe'].isin([600]) &
               data['Nachdruckzeit'].isin([3]) &
               data['Staudruck'].isin([25]) &
               data['Kühlzeit'].isin([15])&
               data['Einspritzgeschwindigkeit'].isin([16])]


T250 = data_subset.loc[data_subset['Düsentemperatur'].isin([250])]
T260 = data_subset.loc[data_subset['Düsentemperatur'].isin([260])]

sns.histplot(data=T250, x=quality_meas,
             color=palette[1],stat='probability',bins=num_bins,
             binwidth=binwidth)
sns.histplot(data=T260, x=quality_meas,
             color=palette[2],stat='probability',bins=num_bins,
             binwidth=binwidth)

axs.set_xlabel(quality_meas)
axs.set_ylabel(None)
fig.tight_layout(pad=0.0, h_pad=None, w_pad=None, rect=None)


# Einfluss Umschaltpunkt#######################################################
# plt.close('all')
palette = sns.color_palette()

fig, axs = plt.subplots() #plt.subplots(2,gridspec_kw={'height_ratios': [1, 1.5]})
fig.set_size_inches((9/2.54,4/2.54))

sns.histplot(data=data, x=quality_meas,stat='probability',
             color=palette[0],bins=num_bins,
             binwidth=binwidth) # Histogramm of all data

data_subset = data.loc[data['Düsentemperatur'].isin([250]) &
               data['Werkzeugtemperatur'].isin([40]) &
               data['Einspritzgeschwindigkeit'].isin([16]) &
               # data['Umschaltpunkt'].isin([14]) &
               data['Nachdruckhöhe'].isin([600]) &
               data['Nachdruckzeit'].isin([3]) &
               data['Staudruck'].isin([25]) &
               data['Kühlzeit'].isin([15])]


U13 = data_subset.loc[data_subset['Umschaltpunkt'].isin([13])]
U14 = data_subset.loc[data_subset['Umschaltpunkt'].isin([14])]

sns.histplot(data=U13, x=quality_meas,
             color=palette[1],stat='probability',bins=num_bins,
             binwidth=binwidth)
sns.histplot(data=U14, x=quality_meas,
             color=palette[2],stat='probability',bins=num_bins,
             binwidth=binwidth)

axs.set_xlabel(quality_meas,fontsize=12)
axs.set_ylabel(None)
fig.tight_layout(pad=0.0, h_pad=None, w_pad=None, rect=None)


########## Scatterplot Qualitätsgrößen & Faktoren #############################

# sns.set_theme(style="ticks")

# quality = ['Gewicht','Durchmesser_innen','Durchmesser_außen',
#                 'Stegbreite_Gelenk','Breite_Lasche','Rundheit_außen']

# quality_short = ['Gewicht','D innen','D außen',
#                 'Stegbr.','Laschenbr.','Rundh. außen']

# factors = ['Düsentemperatur', 'Werkzeugtemperatur',
#        'Einspritzgeschwindigkeit', 'Umschaltpunkt', 'Nachdruckhöhe',
#        'Nachdruckzeit', 'Staudruck', 'Kühlzeit']

# factors_short = ['Düsentemp.', 'Wkz.-Temp.',
#        'Einspr.Geschw.', 'Umschaltpkt.', 'Nachdruckh.',
#        'Nachdruckz.', 'Staudr.', 'Kühlz.']

# plt.close('all')

# fontsize = 9

# grid = sns.pairplot(data_clean.loc[:, quality])
# grid.fig.set_size_inches((10,5.6))


# for i in range(0,len(grid.axes[5,:])):
#     grid.axes[5,i].set_xlabel(quality_short[i],fontsize=fontsize)

# for j in range(0,len(grid.axes[:,0])):
#     grid.axes[j,0].set_ylabel(quality_short[j],fontsize=fontsize)

# grid.fig.tight_layout(pad=0.0, h_pad=None, w_pad=None, rect=None)


# grid = sns.pairplot(data_clean.loc[:, factors])
# grid.fig.set_size_inches((10,5.6))

# for i in range(0,len(grid.axes[7,:])):
#     grid.axes[7,i].set_xlabel(factors_short[i],fontsize=fontsize)

# for j in range(0,len(grid.axes[:,0])):
#     grid.axes[j,0].set_ylabel(factors_short[j],fontsize=fontsize)


# grid.fig.tight_layout(pad=0.0, h_pad=None, w_pad=None, rect=None)














