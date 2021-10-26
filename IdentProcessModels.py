# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:44:37 2021

@author: alexa
"""

import pickle as pkl

from DIM.miscellaneous.PreProcessing import arrange_data_for_ident


cycle1 = pkl.load(open('E:\GitHub\DigitalTwinInjectionMolding\data\Versuchsplan/cycle1.pkl','rb'))


x_names = ['p_wkz_ist','T_wkz_ist','p_inj_ist','Q_Vol_ist','V_Screw_ist']
u_inj_names = ['v_inj_soll']
u_press_names = ['v_inj_soll']
u_cool_names = []


inject,press,cool = arrange_data_for_ident(cycle1,x_names,u_inj_names,
                                           u_press_names,u_cool_names)
