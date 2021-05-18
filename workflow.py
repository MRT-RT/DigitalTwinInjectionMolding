# -*- coding: utf-8 -*-
from sys import path
#path.append(r"C:\Users\LocalAdmin\Documents\casadi-windows-py38-v3.5.5-64bit")

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np

from models import injection_molding, model_structures
from optim.control_optim import MultiStageOptimization
import pickle as pkl
# from miscellaneous import *

''' Define instance of Process Model '''
ProcessModel = injection_molding.ProcessModel()

''' Assume subsystems are first order systems for demonstration purposes '''
dt = 1

# injection_model = model_structures.FirstOrderSystem(dt,'injection_model')
# injection_model.Parameters['a']=-0.9
# injection_model.Parameters['b']=1

injection_model = model_structures.SecondOrderSystem(dt,'injection_model')
injection_model.Parameters={'A':np.array([[0,1],[-1,-1]]),'b':np.array([[0],[1]]),'c':np.array([[1,0]])}

packing_model = model_structures.FirstOrderSystem(dt,'packing_model')
packing_model.Parameters['a']=-0.1
packing_model.Parameters['b']=0.5

cooling_model = model_structures.FirstOrderSystem(dt,'cooling_model')
cooling_model.Parameters['a']=-0.6
cooling_model.Parameters['b']=0


ProcessModel.subsystems = [injection_model,packing_model,cooling_model]

''' Define reference as lambda functions'''

# W1 = lambda param,k: param['h1']+(param['h2']-param['h1'])/(1+np.exp(-2*(k-param['T1'])))
# W2 = lambda param,k: param['h3']+(param['h4']-param['h3'])/(1+np.exp(-2*(k-param['T2'])))



# h1 = np.array([[0]])
# h2 = np.array([[2]])
# T1 = np.array([[30]])
# h3 = np.array([[0]])
# h4 = np.array([[8]])
# T2 = np.array([[120]])

# ProcessModel.ref_params = {'h1': h1, 'h2': h2, 'T1': T1,'h3': h3, 
#                                     'h4': h4, 'T2': T2}


W1 = lambda p,k: p['h1']+(p['h2']-p['h1'])/(1+np.exp(-2*(k-p['T1']))) + p['h2']+(p['h3']-p['h2'])/(1+np.exp(-2*(k-p['T2'])))
W2 = lambda p,k: p['h4']+(p['h5']-p['h4'])/(1+np.exp(-2*(k-p['T3'])))

ProcessModel.reference = [[W1],[W2],[]]

params = {'h1':np.array([[0]]),
          'h2':np.array([[5]]),
          'h3':np.array([[3]]),
          'T1':np.array([[10]]),
          'T2':np.array([[40]]),
          'h4':np.array([[6]]),
          'h5':np.array([[1]]),
          'T3':np.array([[80]])}


ProcessModel.ref_params = params





''' Define switching instances between subsystems '''
ProcessModel.switching_instances = [40,120]



''' Define an (in this case) arbitrary desired target trajectory '''
N=150

# target = cs.sin(np.linspace(0,3*np.pi,N))


target = np.zeros((N,1))
target[0:20]=2
target[20:40]=6
target[40:80]=4
target[80:100]=4
target[100:120]=2
target[120:150]=0

values = MultiStageOptimization(ProcessModel,target)

plt.plot(target,label='soll')
plt.plot(values['X'],label='íst')

ref1 = np.array(W1(values,np.arange(0,150,1))).reshape((N,))
ref2 = np.array(W2(values,np.arange(0,150,1))).reshape((N,))
plt.plot(ref1,label='W1')
plt.plot(ref2,label='W2')
plt.legend()

# ProcessModel.RefTrajectoryInject = {'U1': lambda param,k: param['h1']+(param['h2']-param['h1'])/(1+exp(-2*(k-param['T1']))),
#                         'U2': lambda param,k: param['h3']+(param['h4']-param['h3'])/(1+exp(-2*(k-param['T2'])))}

# ProcessModel.RefTrajectoryPress = {'U1': lambda param,k: param['w1']+(param['w2']-param['w1'])/(1+exp(-2*(k-param['H1']))),
#                         'U2': lambda param,k: param['w3']+(param['w4']-param['w3'])/(1+exp(-2*(k-param['H2'])))}

# W1 = lambda param,k: param['h1']+(param['h2']-param['h1'])/(1+cs.exp(-2*(k-param['T1'])))
# W2 = lambda param,k: param['h3']+(param['h4']-param['h3'])/(1+cs.exp(-2*(k-param['T2'])))

# W3 = lambda param,k: param['w1']+(param['w2']-param['w1'])/(1+cs.exp(-2*(k-param['H1'])))
# W4 = lambda param,k: param['w3']+(param['w4']-param['w3'])/(1+cs.exp(-2*(k-param['H2'])))


# ProcessModel.reference = [[W1,W2],[W3,W4],[]]

# ProcessModel.switching_instances = [40]



# ProcessModel.NumStates = 2


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
N = 100;
x = np.zeros((N,1))
u = np.random.normal(0,1,(N-1,1))

for i in range(1,100):
    x[i] = 0.1*x[i-1] + u[i-1]

# u = [u,None]
# x = [x,None]
    
''' Reestimate Parameters if need be '''
# values = UpdateModelParams(model.ModelInject,u,x,model.ModelParamsInject)

' Decide somehow if old values should be overwritten by new values'''


''' Solve Optimal Control Problem '''

# Ermittle erforderlichen Prozessgrößenverlauf um geforderte Bauteilqualität zu erreichen
# Ergebnis ist reference dictionary mit Referenztrajektorien und Umschaltpunkt





''' Ein Postprocessing bei dem die ermittelten Parameter, die damit verbundenen 
Kosten und der Verlauf über die Zeit angezeigt wird ?'''






 


