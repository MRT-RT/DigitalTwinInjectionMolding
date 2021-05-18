#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:05:48 2020

@author: alexander
"""
from casadi import *

# # System dynamics
# x = MX.sym('x',2)
# u = MX.sym('u',1)

# f1 = x[0]*x[1] + u
# f2 = x[0]

# f = vertcat(f1,f2)

# # Loss function

# L = x[0]**2 + 10*x[1]**2 + u**2

# # Hamiltonian

# lambd = MX.sym('lambd',2,1)

# H = L + lambd.T @ f

# System dynamics
x = MX.sym('x',2)
u = MX.sym('u',1)

lambd = MX.sym('lambd',2,1)

# RHS of differential equation and of 

f1 = x[0]*x[1]  - lambd[0]/2
f2 = x[0]

a1 = 2*x[0] + lambd[0]*x[1] + lambd[1]
a2 = 20*x[1] + lambd[0]*x[0]

# Now implement integration routine for rhs
ode = {'x':vertcat(x,lambd),'ode':vertcat(f1,f2,-a1,-a2)}

options = dict()
options["tf"] = 5

intg = integrator('intg','cvodes',ode,options)

N = 5000

x = [[0,1,3,8.5]]

intg(x0=x[-1])
 
for i in range(N):
    x_new = intg(x0=x[-1])['xf']
    x.append(x_new)

g = cos(x[0]+x[1])+0.5
h = sin(x[0])+0.5

## 1.2

lambd = MX.sym('lambd',2,1) # multiplier for g
nu = MX.sym('nu') # multiplier for h

lag = f+lambd*g+nu*h

lagf = Function('lagf',[x,lambd,nu],[lag])

print(lagf([-0.5,-1.8],2,3)) # Should be 0.875613

## 1.3


QPf = Function('QPf',[x,lambd,nu],[f,g,h,gradient(f,x),hessian(lag,x)[0],jacobian(g,x),jacobian(h,x)])

## 1.4

x0 = [-0.5,-1.8]
lambda0 = 0.
nu0 = 0

# Perform the NLP linearization
[nlp_f,nlp_g,nlp_h,nlp_grad_f,nlp_hess_l,nlp_jac_g,nlp_jac_h] = QPf(x0,lambda0,nu0)

DM.set_precision(16)

print(nlp_hess_l)

## 1.5

H = nlp_hess_l
G = nlp_grad_f
A = vertcat(nlp_jac_g,nlp_jac_h)

lba = vertcat(-nlp_g,-inf)
uba = vertcat(-nlp_g,-nlp_h)

print(H)
print(G)
print(A)
print(lba)
print(uba)

## 1.6

print("H:")
print(H.sparsity()) # 4 nonzeros
H.sparsity().spy()
print("A:")
print(A.sparsity()) # 3 nonzeros

# The difference is between structural sparsity (displayed with 00),
# and sparsity by numerical coincidence.
# Indeed, change lambda0 or nu0 and the Hessian will become dense

## 1.7
qp_struct = {"h":H.sparsity(),"a":A.sparsity()}

solver = conic("solver","qrqp",qp_struct)
print(type(solver)) # casadi.Function

## 1.8
print(solver)
res = solver(h=H,g=G,a=A,lba=lba,uba=uba)
print(res)

dx = res["x"] # should be [-0.02344447381848419, 0.2464226943869885]
lambd = res["lam_a"][0] # should be 0.3785941041969475
nu = res["lam_a"][1] # should be 0.871222132298292

print(dx)
print(lambd)
print(nu)

## 1.9

x = [-0.5,-1.8]
lambd = 0
nu = 0

opts = {"print_iter": False}

qp_struct = {"h":H.sparsity(),"a":A.sparsity()}
solver = conic("solver","qrqp",qp_struct,opts)

for i in range(4):

    # Compute linearizations
    [nlp_f,nlp_g,nlp_h,nlp_grad_f,nlp_hess_l,nlp_jac_g,nlp_jac_h] = QPf(x,lambd,nu)

    # Compose into matrices expected by solver
    H = nlp_hess_l
    G = nlp_grad_f
    A = vertcat(nlp_jac_g,nlp_jac_h)

    lba = vertcat(-nlp_g,-inf)
    uba = vertcat(-nlp_g,-nlp_h)

    # Call solver
    res = solver(h=H,g=G,a=A,lba=lba,uba=uba)

    # Interpret results
    dx = res["x"]
    lambd = res["lam_a"][0]
    nu = res["lam_a"][1]

    # Take a step in decision space
    x = x + dx
    print("x =", x)

## 2.1

# 4 equations, 4 unknowns

# rootfinder x:  [x;lambda;nu]
# rootfinder p: tau (something we may tune)
# rootfinder g: nu*h+tau

# 2.2

x = MX.sym('x',2)

f = x[0]**2+tanh(x[1])**2
g = cos(x[0]+x[1])+0.5
h = sin(x[0])+0.5

lambd = MX.sym('lambd') # multiplier for g
nu = MX.sym('nu') # multiplier for h

lag = f+lambd*g+nu*h

tau = MX.sym('tau')

G = dict()
G["x"] = vertcat(x,lambd,nu)
G["p"] = tau
G["g"] = vertcat(gradient(lag,x),g,nu*h+tau)

rf = rootfinder('rf','newton',G)

print(rf)

## 2.3

x0 = [-0.5,-1.8]
lambd0 = 0.1
nu0 = 0.1

res = rf(x0=vertcat(x0,lambd0,nu0),p=1e-2)
print(res["x"][:2])

## 2.4
res = rf(x0=vertcat(x0,lambd0,nu0),p=1e-6)
print(res["x"[:2]])