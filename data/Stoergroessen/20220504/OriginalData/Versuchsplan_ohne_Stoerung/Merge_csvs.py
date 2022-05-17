#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 17:10:15 2022

@author: alexander
"""
import pandas as pd

quality_filename = 'Messdaten_Verschlusskappe_Versuchsplan_orig_Stgrsn.csv'
weight_filename = 'Gewicht_orig_Strgn.csv'
plan_filename = 'Versuchsplan_Strgn.csv'


df_quality = pd.read_csv(quality_filename,sep=';',index_col=0)
df_weight = pd.read_csv(weight_filename,sep=';',header = None)
df_plan = pd.read_csv(plan_filename,sep=';')


# df_weight.index = df_plan.index

### BRINGE GEWICHTSDATEN IN ERFORLDERLICHES FORMAT ############################

df_weight = df_weight.loc[0:249]                                                # letztes Gewicht doppelt
df_weight.index = range(1,251)
df_weight.index.rename('Zyklusnummer',inplace=True) 
df_weight.rename(columns={0:'Gewicht'},inplace=True) 

### BRINGE VERSUCHSPLAN IN ERFORDERLICHES FORMAT ##############################

df_plan = df_plan.rename(columns={'Düse': 'Düsentemperatur',
                                  'WKZ': 'Werkzeugtemperatur',
                                  'Einspritzgeschw.': 'Einspritzgeschwindigkeit'})
df_plan.index = range(1,26)

col_names = ['Zyklusnummer','Charge','Düsentemperatur','Werkzeugtemperatur',
             'Einspritzgeschwindigkeit','Umschaltpunkt','Nachdruckhöhe',
             'Nachdruckzeit','Staudruck','Kühlzeit','Gewicht','Durchmesser_innen',
             'Durchmesser_außen','Stegbreite_Gelenk','Breite_Lasche','Rundheit_außen']

df_plan_new = pd.DataFrame(data=[],columns=col_names)

df_plan_new['Zyklusnummer'] = range(1,251)
df_plan_new = df_plan_new.set_index('Zyklusnummer')


col = ['Charge','Düsentemperatur', 'Werkzeugtemperatur', 
       'Einspritzgeschwindigkeit','Umschaltpunkt']

for c in df_plan.index:
    df_plan_new.loc[(c-1)*10+1:c*10,col] = df_plan.loc[c][col].values


######### HÄNGE QUALITÄTSDATEN VON MESSPROJEKTOR AN ###########################
col = ['Durchmesser_innen', 'Durchmesser_außen','Stegbreite_Gelenk',
       'Breite_Lasche','Rundheit_außen']

for z in df_quality.index:
    df_plan_new.loc[z,col] = df_quality.loc[z,col]
    
# Folgende Faktoren wurden konstant gehalten
df_plan_new['Nachdruckhöhe'] = 600.0    
df_plan_new['Nachdruckzeit'] = 4.0
df_plan_new['Staudruck'] = 50
df_plan_new['Kühlzeit'] = 20.0

######### HÄNGE QUALITÄTSDATEN VON WAAGE AN ###########################
col = ['Gewicht']

for z in df_weight.index:
    df_plan_new.loc[z,col] = df_weight.loc[z,col]



df_plan_new.to_csv('Versuchsplan_Stgrn.csv',sep=';')










