# -*- coding: utf-8 -*-
from sys import path
#path.append(r"C:\Users\LocalAdmin\Documents\casadi-windows-py38-v3.5.5-64bit")

from casadi import *
import matplotlib.pyplot as plt
import numpy as np

import Modellklassen as Model
from OptimizationTools import *
from miscellaneous import *


''' Model Identification '''
############


''' Model of Injection Moulding Process '''
model = Model.InjectionMouldingMachine()
partmodel = Model.Part()

N=60

h1 = np.array([[1]])
h2 = np.array([[2]])
T1 = np.array([[35]])


model.Führungsgrößenparameter = {'h1': h1, 'h2': h2, 'T1': T1}
model.Führungsgrößen = {'U1': lambda param,k: param['h1']+(param['h2']-param['h1'])/(1+exp(-2*(k-param['T1'])))}



# Define Models for Injection and Pressure Phase

# Construct a CasADi function for the ODE right-hand side
x = MX.sym('x',1) # states: pos_x [m], pos_y [m], vel_x [m/s], vel_y [m/s]
u = MX.sym('u',1) # control force [N]
a1 = MX.sym('a1',1)
a2 = MX.sym('a2',1)

rhs1 = a1*x + u
rhs2 = a2*x + u

# Discrete system dynamics as a CasADi Function
model.ModelInject= Function('ModelInject', [x,u,a1], [rhs1],['x','u','a1'],['rhs1'])
model.ModelPress= Function('ModelPress', [x,u,a2], [rhs2],['x','u','a2'],['rhs2'])

model.ModelParamsInject = {'a1':np.array([[0.9]])}
model.ModelParamsPress = {'a2':np.array([[0.9]])}

model.NumStates = 1


""" Model of Part """

# x = MX.sym('x',1) 
# y = MX.sym('y',1)
# u = MX.sym('u',1) 

# a3 = MX.sym('a3',1)
# c3 = MX.sym('c3',1)

# rhs3 = a3*x + u
# out3 = c3*rhs3

# partmodel.ModelQuality = Function('ModelQuality', [x,u,a3,c3], [rhs3,out3],
#                                   ['x','u','a3','c3'],['rhs3','out3'])

# partmodel.ModelParamsQuality = {'a3':np.array([[0.9]]),'c3':np.array([[2]])}

# U = SingleStageOptimization(partmodel,2,100)


''' Everything from here on needs to run in a loop'''


''' Reestimate Part Quality Model if need be '''




''' Gather Data from Last Shot '''
# N = 100;
# x = np.zeros((N,1))
# u = np.random.normal(0,1,(N-1,1))

# for i in range(1,100):
#     x[i] = 0.1*x[i-1] + u[i-1]

# u = [u,None]
# x = [x,None]
    
''' Reestimate Parameters if need be '''
# values = UpdateModelParams(model.ModelInject,u,x,model.ModelParamsInject)

' Decide somehow if old values should be overwritten by new values'''


''' Solve Optimal Control Problem '''

# Ermittle erforderlichen Prozessgrößenverlauf um geforderte Bauteilqualität zu erreichen
# Ergebnis ist reference dictionary mit Referenztrajektorien und Umschaltpunkt

# Gebe Prozessgrößenverlauf als Referenztrajektorie vor und ermittle erforderliche Maschinenparameter
N=200

reference = {}
reference['Umschaltpunkt'] = 40
reference['data'] = sin(np.linspace(0,3/2*np.pi,N))

values = MultiStageOptimization(model,reference)



''' Ein Postprocessing bei dem die ermittelten Parameter, die damit verbundenen 
Kosten und der Verlauf über die Zeit angezeigt wird ?'''






 


