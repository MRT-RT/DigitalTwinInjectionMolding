#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 09:42:57 2022

@author: alexander
"""
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time

import sys
sys.path.insert(0, "E:\GitHub\DigitalTwinInjectionMolding")
sys.path.insert(0, '/home/alexander/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, 'C:/Users/LocalAdmin/Documents/GitHub/DigitalTwinInjectionMolding/')

# from DIM.models.model_structures import GRU
# from DIM.models.injection_molding import QualityModel
# from DIM.optim.common import BestFitRate
# from DIM.optim.param_optim import parallel_mode
from DIM.miscellaneous.PreProcessing import LoadDynamicData


# Lade Plan
# Finde Chargen in denen starkes transientes Verhalten
# finde chargen in denen schwaches transienstes verhalten
# werte modelle getrennt auf daten aus
# fertig


# path_sys = 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/'
# path_sys = 'C:/Users/LocalAdmin/Documents/GitHub/DigitalTwinInjectionMolding/'
path_sys = '/home/alexander/GitHub/DigitalTwinInjectionMolding/' 
# path_sys = 'E:/GitHub/DigitalTwinInjectionMolding/'

path = path_sys + 'data/Versuchsplan/normalized/'

plan = pkl.load(open(path+'Versuchsplan.pkl','rb'))

charge_low_var = []
charge_high_var = []


for charge in list(range(1,275)):
    plan_sub = plan.loc[plan['Charge']==charge]
    
    std = plan_sub['Durchmesser_innen'].std()
    
    if std > 0.035:
        charge_high_var.append(charge)
    else:
        charge_low_var.append(charge)

