#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 17:10:15 2022

@author: alexander
"""
import pandas as pd
import numpy as np

weight_filename = 'Gewicht_Strgrsn_Umschaltpkt_WKZTemp.csv'
quality_filename = 'Messdaten_Strgrsn_Umschaltpunkt_WKZTemp.csv'

df_weight = pd.read_csv(weight_filename,sep=';',header = None)
df_quality = pd.read_csv(quality_filename,sep=';')


### BRINGE MESSPROJEKT DATEN IN ERFORLDERLICHES FORMAT ########################
df_quality.index = range(251,410)
df_quality.index.rename('Zyklusnummer',inplace=True) 

### BRINGE GEWICHTSDATEN IN ERFORLDERLICHES FORMAT ############################

df_weight.index = range(251,410)
df_weight.index.rename('Zyklusnummer',inplace=True) 
df_weight.rename(columns={0:'Gewicht'},inplace=True) 

### BRINGE VERSUCHSPLAN IN ERFORDERLICHES FORMAT ##############################

col_names = ['Zyklusnummer','Charge','Werkzeugtemperatur',
             'Kühlzeit','Gewicht','Durchmesser_innen',
             'Durchmesser_außen','Stegbreite_Gelenk','Breite_Lasche','Rundheit_außen']

df_plan_new = pd.DataFrame(data=[],columns=col_names)

df_plan_new['Zyklusnummer'] = range(251,410)
df_plan_new = df_plan_new.set_index('Zyklusnummer')



######### HÄNGE QUALITÄTSDATEN VON MESSPROJEKTOR AN ###########################
col = ['Durchmesser_innen', 'Durchmesser_außen','Stegbreite_Gelenk',
       'Breite_Lasche','Rundheit_außen']

for z in df_quality.index:
    df_plan_new.loc[z,col] = df_quality.loc[z,col]
    
# Alle Faktoren wurden konstant gehalten

df_plan_new['Düsentemperatur'] = 250.00
df_plan_new['Werkzeugtemperatur'] = np.nan
df_plan_new['Kühlzeit'] = 20.0

######### HÄNGE QUALITÄTSDATEN VON WAAGE AN ###########################
col = ['Gewicht']

for z in df_weight.index:
    df_plan_new.loc[z,col] = df_weight.loc[z,col]



df_plan_new.to_csv('Strgrsn_Umschaltpkt_WKZTemp.csv',sep=';')










