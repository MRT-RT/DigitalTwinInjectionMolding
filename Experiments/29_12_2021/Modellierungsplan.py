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

dim_c = 2

versuchsplan = pkl.load(open('./data/Versuchsplan/Versuchsplan.pkl','rb'))

c1 = versuchsplan[
    # (versuchsplan['Düsentemperatur']==250) & 
                       (versuchsplan['Werkzeugtemperatur']==40)& 
                       (versuchsplan['Einspritzgeschwindigkeit']==48) & 
                       (versuchsplan['Umschaltpunkt']==14) &
                       (versuchsplan['Nachdruckhöhe']==500) & 
                       (versuchsplan['Nachdruckzeit']==3) & 
                       (versuchsplan['Staudruck']==75) & 
                       (versuchsplan['Kühlzeit']==15)
                       ]['Charge'].unique()

c2 = versuchsplan[(versuchsplan['Düsentemperatur']==250) & 
                       # (versuchsplan['Werkzeugtemperatur']==40)& 
                        (versuchsplan['Einspritzgeschwindigkeit']==48) & 
                       (versuchsplan['Umschaltpunkt']==14) &
                       (versuchsplan['Nachdruckhöhe']==500) & 
                       (versuchsplan['Nachdruckzeit']==3) & 
                       (versuchsplan['Staudruck']==75) & 
                       (versuchsplan['Kühlzeit']==15)
                       ]['Charge'].unique()

c3 = versuchsplan[(versuchsplan['Düsentemperatur']==250) & 
                       (versuchsplan['Werkzeugtemperatur']==40)& 
                       # (versuchsplan['Einspritzgeschwindigkeit']==48) & 
                       (versuchsplan['Umschaltpunkt']==14) &
                       (versuchsplan['Nachdruckhöhe']==500) & 
                       (versuchsplan['Nachdruckzeit']==3) & 
                       (versuchsplan['Staudruck']==75) & 
                       (versuchsplan['Kühlzeit']==15)
                       ]['Charge'].unique()

c4 = versuchsplan[(versuchsplan['Düsentemperatur']==250) & 
                       (versuchsplan['Werkzeugtemperatur']==40)& 
                       (versuchsplan['Einspritzgeschwindigkeit']==48) & 
                       # (versuchsplan['Umschaltpunkt']==14) &
                       (versuchsplan['Nachdruckhöhe']==500) & 
                       (versuchsplan['Nachdruckzeit']==3) & 
                       (versuchsplan['Staudruck']==75) & 
                       (versuchsplan['Kühlzeit']==15)
                       ]['Charge'].unique()

c5 = versuchsplan[(versuchsplan['Düsentemperatur']==250) & 
                       (versuchsplan['Werkzeugtemperatur']==40)& 
                       (versuchsplan['Einspritzgeschwindigkeit']==48) & 
                       (versuchsplan['Umschaltpunkt']==14) &
                       # (versuchsplan['Nachdruckhöhe']==500) & 
                       (versuchsplan['Nachdruckzeit']==3) & 
                       (versuchsplan['Staudruck']==75) & 
                       (versuchsplan['Kühlzeit']==15)
                       ]['Charge'].unique()


c6 = versuchsplan[(versuchsplan['Düsentemperatur']==250) & 
                       (versuchsplan['Werkzeugtemperatur']==40)& 
                       (versuchsplan['Einspritzgeschwindigkeit']==48) & 
                       (versuchsplan['Umschaltpunkt']==14) &
                       (versuchsplan['Nachdruckhöhe']==500) & 
                       # (versuchsplan['Nachdruckzeit']==3) & 
                       (versuchsplan['Staudruck']==75) & 
                       (versuchsplan['Kühlzeit']==15)
                       ]['Charge'].unique()


c7 = versuchsplan[(versuchsplan['Düsentemperatur']==250) & 
                       (versuchsplan['Werkzeugtemperatur']==40)& 
                       (versuchsplan['Einspritzgeschwindigkeit']==48) & 
                       (versuchsplan['Umschaltpunkt']==14) &
                       (versuchsplan['Nachdruckhöhe']==500) & 
                       (versuchsplan['Nachdruckzeit']==3) & 
                       # (versuchsplan['Staudruck']==75) & 
                       (versuchsplan['Kühlzeit']==15)
                       ]['Charge'].unique()

c8 = versuchsplan[(versuchsplan['Düsentemperatur']==250) & 
                       (versuchsplan['Werkzeugtemperatur']==40)& 
                       (versuchsplan['Einspritzgeschwindigkeit']==48) & 
                       (versuchsplan['Umschaltpunkt']==14) &
                       (versuchsplan['Nachdruckhöhe']==500) & 
                       (versuchsplan['Nachdruckzeit']==3) & 
                       (versuchsplan['Staudruck']==75) 
                       # (versuchsplan['Kühlzeit']==15)
                       ]['Charge'].unique()


c9 = versuchsplan[(versuchsplan['Düsentemperatur']==250) & 
                       (versuchsplan['Werkzeugtemperatur']==40)& 
                       (versuchsplan['Einspritzgeschwindigkeit']==48) & 
                       # (versuchsplan['Umschaltpunkt']==14) &
                       (versuchsplan['Nachdruckhöhe']==500) & 
                       # (versuchsplan['Nachdruckzeit']==3) & 
                       (versuchsplan['Staudruck']==75) & 
                        (versuchsplan['Kühlzeit']==15)
                       ]['Charge'].unique()

c10 = versuchsplan[(versuchsplan['Düsentemperatur']==250) & 
                       (versuchsplan['Werkzeugtemperatur']==40)& 
                       (versuchsplan['Einspritzgeschwindigkeit']==48) & 
                       # (versuchsplan['Umschaltpunkt']==14) &
                       (versuchsplan['Nachdruckhöhe']==500) & 
                       # (versuchsplan['Nachdruckzeit']==3) & 
                       (versuchsplan['Staudruck']==75) 
                       # (versuchsplan['Kühlzeit']==15)
                       ]['Charge'].unique()

c11 = versuchsplan[(versuchsplan['Düsentemperatur']==250) & 
                       (versuchsplan['Werkzeugtemperatur']==40)& 
                       (versuchsplan['Einspritzgeschwindigkeit']==48) & 
                       # (versuchsplan['Umschaltpunkt']==14) &
                       # (versuchsplan['Nachdruckhöhe']==500) & 
                       # (versuchsplan['Nachdruckzeit']==3) & 
                       (versuchsplan['Staudruck']==75) & 
                        (versuchsplan['Kühlzeit']==15)
                       ]['Charge'].unique()

c12 = versuchsplan[(versuchsplan['Düsentemperatur']==250) & 
                       (versuchsplan['Werkzeugtemperatur']==40)& 
                       # (versuchsplan['Einspritzgeschwindigkeit']==48) & 
                       # (versuchsplan['Umschaltpunkt']==14) &
                       (versuchsplan['Nachdruckhöhe']==500) & 
                       # (versuchsplan['Nachdruckzeit']==3) & 
                       (versuchsplan['Staudruck']==75) 
                       # (versuchsplan['Kühlzeit']==15)
                       ]['Charge'].unique()

c13 = versuchsplan[(versuchsplan['Düsentemperatur']==250) & 
                       (versuchsplan['Werkzeugtemperatur']==40)
                       # (versuchsplan['Einspritzgeschwindigkeit']==48) & 
                       # (versuchsplan['Umschaltpunkt']==14) &
                       # (versuchsplan['Nachdruckhöhe']==500) & 
                       # (versuchsplan['Nachdruckzeit']==3) & 
                       # (versuchsplan['Staudruck']==75) & 
                       # (versuchsplan['Kühlzeit']==15)
                       ]['Charge'].unique()

plan = []

for i in range(1,14):
    plan.append(list(eval('c'+str(i))))



pkl.dump(plan,open('Modellierungsplan.pkl','wb'))