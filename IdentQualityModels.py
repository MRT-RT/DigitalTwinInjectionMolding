# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:44:37 2021

@author: alexa
"""

import pickle as pkl

from DIM.miscellaneous.PreProcessing import arrange_data_for_ident











# Load Versuchsplan to find cycles that should be considered for modelling

data = pkl.load(open('data/Versuchsplan/Versuchsplan.pkl','rb'))

cycles_train = data.loc[data['Düsentemperatur'].isin([250]) & 
               data['Werkzeugtemperatur'].isin([40]) &
               ~data['Einspritzgeschwindigkeit'].isin([32]) &
               ~data['Umschaltpunkt'].isin([13.5]) &
               ~data['Nachdruckhöhe'].isin([550]) &
               ~data['Nachdruckzeit'].isin([4]) &
               ~data['Staudruck'].isin([50]) &
               ~data['Kühlzeit'].isin([17.5])].index.values

cycles_val = data.loc[data['Düsentemperatur'].isin([250]) & 
               data['Werkzeugtemperatur'].isin([40])].index.values
               # data['Umschaltpunkt'].isin([14]) &
               # data['Nachdruckhöhe'].isin([600]) &
               # data['Nachdruckzeit'].isin([3]) &
               # data['Staudruck'].isin([25]) &
               # data['Kühlzeit'].isin([15])]
               

cycles_val = np.setxor1d(cycles_train,cycles_val)






cycle1 = pkl.load(open('E:\GitHub\DigitalTwinInjectionMolding\data\Versuchsplan/cycle1.pkl','rb'))


x_names = ['p_wkz_ist','T_wkz_ist','p_inj_ist','Q_Vol_ist','V_Screw_ist']
u_inj_names = ['v_inj_soll']
u_press_names = ['v_inj_soll']
u_cool_names = []



# Choose measured process variables (is)
# Keep temperatures out of the equation for now

# Predict quality measurements : Gewicht, Durchmesser_innen

inject,press,cool = arrange_data_for_ident(cycle1,x_names,u_inj_names,
                                           u_press_names,u_cool_names)
