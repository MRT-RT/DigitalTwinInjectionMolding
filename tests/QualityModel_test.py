# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 11:00:35 2021

@author: alexa
"""

# -*- coding: utf-8 -*-
import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

sys.path.append('E:/GitHub/DigitalTwinInjectionMolding/')
sys.path.append('C:/Users/LocalAdmin/Documents/GitHub/DigitalTwinInjectionMolding/')

from DIM.models.model_structures import GRU
from DIM.models.injection_molding import QualityModel
from DIM.optim.control_optim import QualityMultiStageOptimization

# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'DIM'))

# # import mymodule

# from models.model_structures import QualityModel
# from models.model_structures import GRU
# from optim.control_optim import SingleStageOptimization


# import matplotlib.pyplot as plt
# import numpy as np


GRU1 = GRU(dim_u=1,dim_c=2,dim_hidden=2,dim_out=2,name='GRU1')
GRU2 = GRU(dim_u=1,dim_c=2,dim_hidden=5,dim_out=2,name='GRU2')


QualMod = QualityModel()
QualMod.subsystems = [GRU1,GRU2,GRU2]
QualMod.switching_instances = [200,100,150]

target = np.ones((2,1))

result = QualityMultiStageOptimization(QualMod,target)


c0 = np.zeros((2,1))

sim_res = QualMod.Simulation(c0,result['U'].reshape((-1,1)))




plt.plot(result['U'])
plt.plot(sim_res[1])

