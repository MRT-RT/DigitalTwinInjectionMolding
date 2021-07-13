# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 11:00:35 2021

@author: alexa
"""

from models.injection_molding import QualityModel
from models.model_structures import GRU
from optim.control_optim import SingleStageOptimization


import matplotlib.pyplot as plt
import numpy as np


GRU = GRU(dim_u=1,dim_c=2,dim_hidden=10,dim_out=1,name='QualityGRU')

# u = np.ones((100,1))
# 

# y = GRU.Simulation(c0, u)


QualModel = QualityModel()
QualModel.model = GRU
QualModel.N = 100


result = SingleStageOptimization(QualModel,5)


u = result['U'].reshape(-1,1)
c0 = np.zeros((2,1))

y = GRU.Simulation(c0, u)

plt.figure()
plt.plot(u)
plt.plot(y)