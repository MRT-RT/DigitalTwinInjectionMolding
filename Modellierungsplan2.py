# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:44:37 2021

@author: alexa
"""

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from DIM.miscellaneous.PreProcessing import arrange_data_for_ident, eliminate_outliers

from DIM.models.model_structures import GRU,LSTM
from DIM.models.injection_molding import QualityModel
from DIM.optim.param_optim import ModelTraining, HyperParameterPSO
from DIM.miscellaneous.PreProcessing import LoadData

versuchsplan = pkl.load(open('./data/Versuchsplan/Versuchsplan.pkl','rb'))



c411 = versuchsplan['Charge'].unique()

plan = []

for i in [411]:
    plan.append(list(eval('c'+str(i))))



pkl.dump(plan,open('Modellierungsplan_all.pkl','wb'))