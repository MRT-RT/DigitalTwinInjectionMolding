# -*- coding: utf-8 -*-
import casadi as cs
from .activations import *



def Eval_FeedForward_NN(input, NN,NN_act):
    out = []
    for l in range(0,len(NN)):

        if l == 0:
            out.append(NN_layer(input,cs.horzcat(NN[l][0],
                               NN[l][1]),NN[l][2],NN_act[l]))
        else:
            out.append(NN_layer(out[-1],NN[l][0],NN[l][1],NN_act[l]))    
    
    return out


def NN_layer(input,weights,bias,nonlinearity):
    '''
    Calculates activation of a neural network layer        

    Parameters
    ----------
    input : TYPE
        DESCRIPTION.
    weights : TYPE
        DESCRIPTION.
    bias : TYPE
        DESCRIPTION.
    nonlinearity : TYPE
        DESCRIPTION.

    Returns
    -------
    y : TYPE
        DESCRIPTION.

    '''

    if nonlinearity == 0:
        nonlin = identity    
    if nonlinearity == 1:
        nonlin = cs.tanh
    elif nonlinearity == 2:
        nonlin = logistic
    elif nonlinearity == 3:
        nonlin  = ReLu
            
    net = cs.mtimes(weights,input) + bias

    return nonlin(net)