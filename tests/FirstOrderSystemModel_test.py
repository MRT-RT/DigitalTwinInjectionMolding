# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 12:48:12 2021

@author: alexa
"""

# import os
# print (os.getcwd())


import numpy as np
import matplotlib.pyplot as plt

from models.model_structures import SecondOrderSystem

N = 100

u = np.ones((N,1))

PT2 = SecondOrderSystem(1,'injection_model')
PT2.Parameters['A']=np.array([[0,1],[-1,-1]])
PT2.Parameters['b']=np.array([[0],[1]])
PT2.Parameters['c']=np.array([[1,0]])

x_sim,y_sim = PT2.Simulation(np.array([[0],[0]]), u)

y_sim = np.array(y_sim)

plt.plot(y_sim)