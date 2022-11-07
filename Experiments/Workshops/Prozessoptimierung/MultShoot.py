# -*- coding: utf-8 -*-

import casadi as cs
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dyn_model = pkl.load(open('dyn_model.pkl','rb') )

f_model = dyn_model['f_model']
A_opt = dyn_model['A_opt']
B_opt = dyn_model['B_opt']
C_opt = dyn_model['C_opt']

plt.close('all')

# %%

U_init = np.array([0.09941283, 0.05073077, 0.07485718, 0.08713247, 0.09337803,
       0.0965557 , 0.09817248, 0.09899508, 0.0])


x_sim = [np.zeros((2,1))]
y_sim = []

for k in range(U_init.shape[0]):
    pred = f_model(u=U_init[k],x=x_sim[-1],A=A_opt,B=B_opt,C=C_opt)
    
    x_sim.append(pred['x_new'])
    y_sim.append(pred['y'])

y_sim = np.vstack(y_sim)
x_sim = np.hstack(x_sim)

fig,ax = plt.subplots(1,1)

ax.plot(x_sim.T,marker='o')
ax.plot(U_init,marker='x')


# %%

# opti = cs.Opti()

# N = 10

# U = opti.variable(N-1,1)
# S = opti.variable(5,2)

# y_est = []
# x_est = [S[0,:].T]

# k = 0

# for i in range(S.shape[0]-1):

#     pred = f_model(u=U[k],x=S[i],A=A_opt,B=B_opt,C=C_opt)
#     x_est.append(pred['x_new'])
#     y_est.append(pred['y'])
    
#     k = k + 1
    
#     pred = f_model(u=U[k],x=x_est[-1],A=A_opt,B=B_opt,C=C_opt)
#     x_est.append(pred['x_new'])
#     y_est.append(pred['y'])
    
#     opti.subject_to(x_est[-1] == S[i+1,:].T)
    
#     k = k + 1 
    
# pred = f_model(u=U[k],x=S[4],A=A_opt,B=B_opt,C=C_opt)
# x_est.append(pred['x_new'])
# y_est.append(pred['y'])    
    
# # opti.subject_to(S[0,:].T == np.array([[0],[0]]))    

# # Listen in Arrays konvertieren
# y_est = cs.vcat(y_est)
# x_est = cs.hcat(x_est)

# %%

opti = cs.Opti()

N = 10

U = opti.variable(N-1,1)
S = opti.variable(2,N)

y_est = []
x_est = [S[:,0]]

for k in range(U.shape[0]):
    pred = f_model(u=U[k],x=S[-1],A=A_opt,B=B_opt,C=C_opt)
    
    x_est.append(pred['x_new'])
    
    opti.subject_to(x_est[-1] == S[:,k+1])

    y_est.append(pred['y'])

opti.subject_to(x_est[0] == np.zeros((2,1)))    

# Listen in Arrays konvertieren
y_est = cs.vcat(y_est)
x_est = cs.hcat(x_est)
# %%

# L = cs.sumsqr(x_sim[:,0::2].T-S)
# L = cs.sumsqr(x_sim-S)
L = cs.sumsqr((y_sim - y_est))

opti.minimize(L)

opti.solver('ipopt')

# opti.set_initial(S,X_init[0::2,:])
# opti.set_initial(S,X_init)

# opti.set_initial(U,U_init)

sol = opti.solve()

U_opt = sol.value(U)
X_opt = sol.value(S)


# %% 

x_opt = [X_opt[:,0]]
y_opt = []

for k in range(U_opt.shape[0]):
    pred = f_model(u=U_opt[k],x=x_opt[-1],A=A_opt,B=B_opt,C=C_opt)
    
    x_opt.append(pred['x_new'])
    y_opt.append(pred['y'])

# opti.subject_to(S[0,:].T == np.array([[0],[0]]))    

# Listen in Arrays konvertieren
y_opt = cs.vcat(y_opt)
x_opt = cs.hcat(x_opt)



fig2,ax2 = plt.subplots(1,1)
ax2.plot(x_opt.T,marker='d')
ax2.plot(X_opt.T,marker='o')
ax2.plot(U_opt.T,marker='x')







# plt.plot(x_sim.T)
