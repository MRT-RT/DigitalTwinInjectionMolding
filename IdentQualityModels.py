# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:44:37 2021

@author: alexa
"""

import pickle as pkl
import numpy as np

from DIM.miscellaneous.PreProcessing import arrange_data_for_ident

from DIM.models.model_structures import GRU
from DIM.models.injection_molding import QualityModel
from DIM.optim.param_optim import ModelParameterEstimation



# # Load Versuchsplan to find cycles that should be considered for modelling
# data = pkl.load(open('data/Versuchsplan/Versuchsplan.pkl','rb'))

# cycles = data.loc[data['Düsentemperatur'].isin([250]) & 
#                data['Werkzeugtemperatur'].isin([40]) &
#                ~data['Einspritzgeschwindigkeit'].isin([32]) &
#                ~data['Umschaltpunkt'].isin([13.5]) &
#                ~data['Nachdruckhöhe'].isin([550]) &
#                ~data['Nachdruckzeit'].isin([4]) &
#                ~data['Staudruck'].isin([50]) &
#                ~data['Kühlzeit'].isin([17.5])].index.values

# # Remove first two cycles of each charge. Use remaining 6 for training and 2 
# # for validation
# cycles_train = np.hstack([np.arange(2,cycles[-1],10),
#                        np.arange(3,cycles[-1],10),
#                        np.arange(4,cycles[-1],10),
#                        np.arange(5,cycles[-1],10),
#                        np.arange(6,cycles[-1],10),
#                        np.arange(7,cycles[-1],10)])

# cycles_val = np.hstack([np.arange(8,cycles[-1],10),
#                         np.arange(9,cycles[-1],10)])


# # Load cycle data, check if usable, convert to numpy array


cycle1 = pkl.load(open('data/Versuchsplan/cycle1.pkl','rb'))


x_names = ['Durchmesser_innen']
u_inj_names = ['p_wkz_ist','T_wkz_ist','p_inj_ist','Q_Vol_ist','V_Screw_ist']
u_press_names = u_inj_names
u_cool_names = u_inj_names

# Choose measured process variables (is)
# Keep temperatures out of the equation for now

# Predict quality measurements : Gewicht, Durchmesser_innen

inject,press,cool = arrange_data_for_ident(cycle1,x_names,u_inj_names,
                                           u_press_names,u_cool_names)



injection_model = GRU(dim_u=1,dim_c=10,dim_hidden=10,dim_out=1,name='inject')
press_model = GRU(dim_u=1,dim_c=10,dim_hidden=10,dim_out=1,name='press')
cool_model = GRU(dim_u=1,dim_c=10,dim_hidden=10,dim_out=1,name='cool')

quality_model = QualityModel()
quality_model.subsystems = [injection_model,press_model,cool_model]

quality_model.switching_instances =[14,339]

# u = cycle1['p_inj_ist'].values.reshape(-1,1)
# c,y = quality_model.Simulation(np.zeros((10,1)),u)

# data = {'u_train': cycle1['p_inj_ist'].values.reshape(-1,1,1),
#         'y_train': cycle1['Durchmesser_innen'].values.reshape(-1,1,1),
#         'init_state_train': np.zeros((10,1,1))}


# values = ModelParameterEstimation(quality_model,data,p_opts=None,s_opts=None)



















