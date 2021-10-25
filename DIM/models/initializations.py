# -*- coding: utf-8 -*-
import numpy as np

# def XavierInitialization(W_shape,b_shape=None,n):
#     '''
#     Routine for Xavier initialization

#     Parameters
#     ----------
#     W : numpy-array
#         n x n-1 vatrix with weights mapping from layer l-1 with n-1 neurons to layer l
#         with n neurons.
#     b : numpy-array
#         n x 1 vector containing biases of l-th layer with n neurons.
#     n : int
#         Number of neurons in previous layer, i.e. layer l-1.

#     Returns
#     -------
#     Wx: numpy-array
#         initialized weights
#     bx: numpy-array
#         initialized biases
#     '''    
    
#     lim = np.sqrt(3)/np.sqrt(n)
    
#     Wx = np.random.uniform(low=-lim, high=lim, size=W_shape)
    
#     if b is not None:
#         bx = np.zeros(b_shape)
#     else:
#         bx = None
#     return Wx,bx
def RandomInitialization(dimensions,bias=False):
    
    return np.random.rand(*dimensions)


def XavierInitialization(dimensions,bias=False):
    '''
    Routine for Xavier initialization

    Parameters
    ----------
    dimensions: tuple
        description
    bias : boolean
        description

    Returns
    -------
    W: numpy-array
        initialized weights
    b: numpy-array
        initialized biases
    '''    
    
    if 0 in dimensions:
        return np.array([]).reshape(dimensions)
    
    if bias == True:
        b = np.zeros((dimensions[0],1))
        return b
    
    else:
        lim = np.sqrt(3)/np.sqrt(dimensions[1])
        W = np.random.uniform(low=-lim, high=lim, size=dimensions)
        return W

def HeInitialization(dimensions,bias=False):
    '''
    Routine for He initialization

    Parameters
    ----------
    dimensions: tuple
        description
    bias : boolean
        description

    Returns
    -------
    W: numpy-array
        initialized weights
    b: numpy-array
        initialized biases
    '''    
    
    if 0 in dimensions:
        return np.array([]).reshape(dimensions)
    
    if bias == True:
        b = np.zeros((dimensions[0],1))
        return b
    
    else:
        var = 2/dimensions[1]
        W = np.random.normal(loc=0, scale=np.sqrt(var), size=dimensions)
        return W
