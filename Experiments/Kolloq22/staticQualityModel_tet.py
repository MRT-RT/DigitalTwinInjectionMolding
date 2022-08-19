# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 17:33:50 2022

@author: LocalAdmin
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import pickle as pkl

import sys


path_dim = Path.cwd().parents[1]
sys.path.insert(0, path_dim.as_posix())

from DIM.miscellaneous.PreProcessing import LoadFeatureData
from functions import estimate_polynomial
from DIM.models.model_structures import Static_MLP
from DIM.models.injection_molding import staticQualityModel
from DIM.optim.param_optim import ModelTraining, static_mode



DoubleExpResults = pkl.load(open('Temp_Models.mdl','rb'))

setpoints = DoubleExpResults['setpoints']
Temp_Models = DoubleExpResults['Temp_Models']


setpoints_lab = ['Düsentemperatur', 'Werkzeugtemperatur', 
                 'Einspritzgeschwindigkeit', 'Umschaltpunkt', 'Nachdruckhöhe',
                 'Nachdruckzeit', 'Staudruck','Kühlzeit']
T = ['T_wkz_0']
Di = ['Durchmesser_innen']

set_model = Static_MLP(8,1,5,setpoints_lab,
                       list(Temp_Models[1].Parameters.keys()),'setpnt')


QM = staticQualityModel(set_model,Temp_Models,setpoints,'QM')