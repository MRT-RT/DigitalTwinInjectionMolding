import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

import sys


sys.path.append('C:/Users/LocalAdmin/Documents/GitHub/DigitalTwinInjectionMolding/')
sys.path.append('E:/GitHub/DigitalTwinInjectionMolding/')
sys.path.append('/home/alexander/GitHub/DigitalTwinInjectionMolding/')


from DIM.miscellaneous.PreProcessing import arrange_data_for_ident

from DIM.models.model_structures import GRU
from DIM.models.injection_molding import QualityModel
from DIM.optim.param_optim import ModelTraining

cycles = []

for i in range(1,11):
    cycles.append(pkl.load(open('../data/Versuchsplan/cycle'+str(i)+'.pkl','rb')))

cycles_train = cycles[2:7]
cycles_val = cycles[7:10]

q_lab = ['Durchmesser_innen']
x_lab= [['p_wkz_ist','T_wkz_ist','p_inj_ist','Q_Vol_ist','V_Screw_ist']]

data_train,init_train,switch_train = arrange_data_for_ident(cycles_train,q_lab,x_lab,'quality')
data_val,init_val,switch_val = arrange_data_for_ident(cycles_val,q_lab,x_lab,'quality')

data_train[:][q_lab]