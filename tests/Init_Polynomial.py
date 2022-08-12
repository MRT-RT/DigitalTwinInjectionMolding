# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 12:45:17 2022

@author: Alexander Rehmer
"""

import sys
import pandas as pd

sys.path.insert(0, "C:/Users/LocalAdmin/Documents/GitHub/DigitalTwinInjectionMolding/")

from DIM.models.model_structures import PolynomialModel
from DIM.optim.param_optim import static_mode


# Create instance of model
PolyModel =  PolynomialModel(dim_u=3,dim_out=1,degree_n=5,interaction=True,
                             u_label=['u1','u2','u3'],y_label=['y'],
                             name='poly')

# Create some random test data 
data = pd.DataFrame(data=[[1,2,3]],columns=['u1','u2','u3'])

# Evaluate model on test data to see if model is implemented properly
_,y_est = static_mode(PolyModel, data)

print('Result is ' + str(y_est.loc[0,'y']))




import casadi as cs


u = cs.MX.sym('u',3,1)
w = cs.MX.sym('W_in',3,1,5) 

y = []

for wi in w:
    y.append( cs.mtimes(wi.T,u) )

y = cs.vertcat(*y)

y = cs.sum1(y)
