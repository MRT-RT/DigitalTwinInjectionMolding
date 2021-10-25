# -*- coding: utf-8 -*-
import casadi as cs
import numpy as np


def logistic(x):
    
    y = 0.5 + 0.5 * cs.tanh(0.5*x)

    return y

def ReLu(x):
    
    # y = cs.horzcat(np.zeros(x.shape),x)
    
    # y_relu = []
    # #
    # for row in range(0,y.shape[0]):
    #     y_relu.append(cs.mmax(y[row,:]))
        
    # y = cs.vcat(y_relu)  
    
    y = (cs.fabs(x)+x)/2
    
    return y


def RBF(x,c,w):
    d = x-c    
    e = - cs.mtimes(cs.mtimes(d.T,cs.diag(w)**2),d)
    y = cs.exp(e)
    
    return y

def identity(x):
    return x