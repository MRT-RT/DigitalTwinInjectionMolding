#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 08:46:01 2022

@author: alexander
"""
import casadi as cs
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

data_train = pkl.load(open('feat_data_train.pkl','rb'))
data_val= pkl.load(open('feat_data_val.pkl','rb'))


W_h = cs.MX.sym('W_h',1,8)
b_h = cs.MX.sym('b_h',1,1)

W_o = theta1 = cs.MX.sym('W_o',1,1)
b_o = cs.MX.sym('b_o',1,1)


theta = cs.vcat([W_h.reshape((-1,1)),b_h,W_o.reshape((-1,1)),b_o])

u = cs.MX.sym('u',8,1)

h = cs.tanh(cs.mtimes(W_h,u)+b_h)
y = cs.mtimes(W_o,h)+b_o



f_model = cs.Function('f_model',[u,theta],[y],['u','theta'],['y'])

L = 0

u_label=['T_wkz_0', 'T_wkz_max', 't_Twkz_max', 'T_wkz_int', 'p_wkz_max','p_wkz_int', 'p_wkz_res', 't_pwkz_max']

y_label = ['Durchmesser_innen']

for k in data_train.index:
    u_k = data_train.loc[k][u_label].values.reshape((8,))
    
    y_k = data_train.loc[k][y_label]
    
    y_hat = f_model(u=u_k, theta=theta)['y']
    L = L + 0.5*(y_hat - y_k)**2
    
nlp = {'x':theta, 'f':L}
S = cs.nlpsol('S', 'ipopt', nlp)

theta_init = np.random.normal(0,1,(11,1))

r=S(x0=theta_init)

theta_opt = r['x']