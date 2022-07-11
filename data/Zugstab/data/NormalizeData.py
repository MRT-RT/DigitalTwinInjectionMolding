#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 10:23:48 2022

@author: alexander
"""

import pickle as pkl
import os

import sys
sys.path.insert(0, "E:\GitHub\DigitalTwinInjectionMolding")
sys.path.insert(0, 'C:/Users/LocalAdmin/Documents/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, "/home/alexander/GitHub/DigitalTwinInjectionMolding/")

from DIM.miscellaneous.PreProcessing import find_switches


# Load cycle with respect to which all data are normalized
norm_cycle = pkl.load(open('cycle210.pkl','rb'))

y_qual = ['Gewicht', 'OK/N.i.O.', 'Durchmesser_innen',
       'Durchmesser_außen', 'Stegbreite_Gelenk', 'Breite_Lasche',
       'Rundheit_außen', 'Dicke', 'Mittenrauwert', 'Quadratischer Mittelwert',
       'Rautiefe', 'Höhe', 'Breite', 'E-Modul', 'Zugfestigkeit',
       'Dehnung bei Zugfestigkeit', 'Maximalspannung', 'Bruchspannung', 
       'Maximalkraft','Dehnung bei Maximalspannung', 'Bruchdehnung',
       'Anfangsfläche']

rest = ['Q_Vol_ist', 'V_Screw_ist', 'p_wkz_ist', 'T_wkz_ist', 'p_inj_soll',
       'p_inj_ist', 'Q_inj_soll', 'T_zyl1_ist', 'T_zyl2_ist', 'T_zyl3_ist',
       'T_zyl4_ist', 'T_zyl5_ist', 'V_um_ist', 'p_um_ist', 'p_inj_max_ist', 
       'T_nozz_soll', 'T_wkz_soll', 'v_inj_soll', 'Umschaltpunkt',
       'Nachdruckhöhe', 'Staudruck']

# Calculate quantities for normalization
mean_y = norm_cycle[y_qual].mean()                                              # This normalization was formerly used for quality models
min_rest = norm_cycle[rest].min()
max_rest = norm_cycle[rest].max()

min_rest[max_rest-min_rest==0]=0                                                # if signal is constant, set minimum to 0 to avoid division by zero    

cycle_files = os.listdir()
cycle_files.remove('NormalizeData.py')
cycle_files.remove('NormalizeData_minmax.py')
cycle_files.remove('Versuchsplan.pkl')
cycle_files.remove('normalized')
cycle_files.remove('normalized_minmax')

for file in cycle_files:
    c = pkl.load(open(file,'rb'))

    t1,_,_ = find_switches(c)
    c.loc[t1]['p_inj_soll']=700.00
        
    c[y_qual] = (c[y_qual]-mean_y)+1
    
    c[rest] = (c[rest]-min_rest)/(max_rest-min_rest)
    
    pkl.dump(c,open('./normalized/'+file,'wb'))