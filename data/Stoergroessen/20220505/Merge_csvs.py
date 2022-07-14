#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 17:10:15 2022

@author: alexander
"""
import pandas as pd
import numpy as np

weight_filename = 'Gewicht_Strgn_WKZ_Temp.csv'
quality_filename = 'Messdaten_Verschlusskappe_Strgn_WKZ_Temp.csv'

df_weight = pd.read_csv(weight_filename,sep=';',header = None)
df_quality = pd.read_csv(quality_filename,sep=';')

df_quality = df_quality.rename(columns={'laufenden Zähler':'Zyklusnummer'})
df_quality = df_quality.set_index('Zyklusnummer')

# Teil 40 und 105 sind verschwunden, daher Index anpassen
idx = list(range(1,143)) 
idx.remove(40)
idx.remove(105)

### BRINGE MESSPROJEKT DATEN IN ERFORLDERLICHES FORMAT ########################

df_quality.drop(index=[137,138,142],inplace=True)

# df_quality.index = range(1,140)

df_quality.index.rename('Zyklusnummer',inplace=True) 

### BRINGE GEWICHTSDATEN IN ERFORLDERLICHES FORMAT ############################

df_nan = pd.DataFrame(data=[], columns=['Gewicht'], index=[136,137,138])
df_weight.rename(columns={0:'Gewicht'},inplace=True) 
df_weight = pd.concat([df_weight,df_nan])

df_weight.index = idx
df_weight.index.rename('Zyklusnummer',inplace=True) 


### BRINGE VERSUCHSPLAN IN ERFORDERLICHES FORMAT ##############################

col_names = ['Zyklusnummer','Charge','Werkzeugtemperatur',
             'Kühlzeit','Gewicht','Durchmesser_innen',
             'Durchmesser_außen','Stegbreite_Gelenk','Breite_Lasche','Rundheit_außen']

df_plan_new = pd.DataFrame(data=[],columns=col_names)

df_plan_new['Zyklusnummer'] = idx
df_plan_new = df_plan_new.set_index('Zyklusnummer')



######### HÄNGE QUALITÄTSDATEN VON MESSPROJEKTOR AN ###########################
col = ['Durchmesser_innen', 'Durchmesser_außen','Stegbreite_Gelenk',
       'Breite_Lasche','Rundheit_außen']

for z in df_quality.index:
    df_plan_new.loc[z,col] = df_quality.loc[z,col]
    
# Alle Faktoren wurden konstant gehalten

df_plan_new['Düsentemperatur'] = 250.00

df_plan_new.loc[1:20,'Werkzeugtemperatur'] = 35
df_plan_new.loc[21:50,'Werkzeugtemperatur'] = 30 
df_plan_new.loc[51:80,'Werkzeugtemperatur'] = 33
df_plan_new.loc[81:110,'Werkzeugtemperatur'] = 35
df_plan_new.loc[111:119,'Werkzeugtemperatur'] = 38
df_plan_new.loc[120:136,'Werkzeugtemperatur'] = 37
df_plan_new.loc[137:,'Werkzeugtemperatur'] = 35 


df_plan_new['Kühlzeit'] = 20.0

df_plan_new['Charge'] = 1

######### HÄNGE QUALITÄTSDATEN VON WAAGE AN ###########################
col = ['Gewicht']

for z in df_weight.index:
    df_plan_new.loc[z,col] = df_weight.loc[z,col]


print('fin')
# df_plan_new.to_csv('Strgrsn_T_wkz.csv',sep=';')










