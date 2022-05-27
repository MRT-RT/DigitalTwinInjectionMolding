#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 17:10:15 2022

@author: alexander
"""
import pandas as pd
import numpy as np

weight_filename = 'Gewicht_Rezyklat.csv'
quality_filename = 'Rezyklat_Messprojektor.csv'

df_weight = pd.read_csv(weight_filename,sep=';',header = None)
df_quality = pd.read_csv(quality_filename,sep=';')


# Teil 40 und 105 sind verschwunden, daher Index anpassen
# idx = list(range(1,142)) 
# idx.remove(40)
# idx.remove(105)

### BRINGE MESSPROJEKT DATEN IN ERFORLDERLICHES FORMAT ########################

# df_quality.index = range(1,140)

df_quality.index = range(1,131)
df_quality.index.rename('Zyklusnummer',inplace=True) 

# Nicht vermessbare Teile, lösche Doppelungen

idx_del = [60,61,80,81,92,93,114,115]
df_quality.drop(index=idx_del,inplace=True)


# Reindiziere
idx = list(range(1,60)) + list(range(61,79)) + list(range(80,90)) + list(range(91,111)) + list(range(112,127))

df_quality.index = idx



# df_quality.drop(index=[61,81,93,115],inplace=True)
### BRINGE GEWICHTSDATEN IN ERFORLDERLICHES FORMAT ############################

df_weight.rename(columns={0:'Gewicht'},inplace=True) 

df_weight.index = range(1,127)
df_weight.index.rename('Zyklusnummer',inplace=True) 


idx_del = [60,79,90,111]
df_weight.drop(index=idx_del,inplace=True)


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
df_plan_new['Werkzeugtemperatur'] = 35
df_plan_new['Kühlzeit'] = 20.0



######### HÄNGE QUALITÄTSDATEN VON WAAGE AN ###########################
col = ['Gewicht']

for z in df_weight.index:
    df_plan_new.loc[z,col] = df_weight.loc[z,col]



df_plan_new.to_csv('Strgrsn_Rezyklat.csv',sep=';')










