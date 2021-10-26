# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:46:17 2021

@author: alexa
"""

import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt

data = pkl.load(open('data/Versuchsplan/Versuchsplan.pkl','rb'))

# quality_meas = 'Gewicht'
# quality_meas = 'Durchmesser_innen'
# quality_meas = 'Durchmesser_außen'
# quality_meas = 'Stegbreite_Gelenk'
# quality_meas = 'Breite_Lasche'
# quality_meas = 'Rundheit_außen'


num_bins = 30
binwidth = 0.01


# Einfluss Einspritzgeschwindigkeit ##########################################
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

# Einfluss Düsentemperatur ###################################################
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


# Einfluss Umschaltpunkt######################################################
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

axs.set_xlabel(quality_meas)
axs.set_ylabel(None)
fig.tight_layout(pad=0.0, h_pad=None, w_pad=None, rect=None)

















