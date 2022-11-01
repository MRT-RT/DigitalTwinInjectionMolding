# -*- coding: utf-8 -*-

from casadi import *

N = 4

F = lambda x,u: x**2+u

## 1.1

opti = casadi.Opti()

X = opti.variable(N+1)
U = opti.variable(N)

for k in range(N):
  opti.subject_to(F(X[k],U[k])==X[k+1])


opti.subject_to(X[0]==2)
opti.subject_to(X[1:-1]>=0)
opti.subject_to(X[-1]==3)

opti.minimize(sumsqr(U)+sumsqr(X))

opti.solver('ipopt')
sol = opti.solve()

Usol_MS = sol.value(U) # should be [-2.7038;-0.5430;0.2613;0.5840]
print(Usol_MS)

opti_MS = opti
U_MS = U
sol_MS = sol


## 1.2

opti = casadi.Opti()

U = opti.variable(N)

x = 2
X = [x]
for k in range(N):
  x = F(x,U[k])
  X.append(x)

X = hcat(X)

opti.subject_to(X[1:-1]>=0)
opti.subject_to(X[-1]==3)

opti.minimize(sumsqr(U)+sumsqr(X))

opti.solver('ipopt')
sol = opti.solve()

Usol_SS = sol.value(U) # should be [-2.7038;-0.5430;0.2613;0.5840]
print(Usol_SS)

opti_SS = opti
U_SS = U
sol_SS = sol

## 1.3


print('MS Hessian')
hessian(opti_MS.f,opti_MS.x)[0].sparsity().spy()
# Only diagonal elements: no coupling between x or u,
# on coupling between control intervals

print('MS jacobian')
jacobian(opti_MS.g,opti_MS.x).sparsity().spy()

# Dynamic constraints: coupling between x_k+1 and x_k and u_k -> banded
# Path constraints: only dependent on x-> diagonal

print('SS Hessian')
hessian(opti_SS.f,opti_SS.x)[0].sparsity().spy()

# Smaller than MS Hessian
# Objective depends in a very nonlinear way on all u -> dense

print('SS jacobian')
jacobian(opti_SS.g,opti_SS.x).sparsity().spy()


# x_k depends on all u_j with j<k -> triangular

## 1.4

opti = casadi.Opti()

x = opti.variable()

opti.subject_to(x==2)

U = []
X = [x]
for k in range(N):
  u = opti.variable()
  xp = opti.variable()
  opti.subject_to(F(x,u)==xp)
  if k<N-1:
    opti.subject_to(xp>=0)
  U.append(u)
  x = xp
  X.append(x)

opti.subject_to(x==3)

U = hcat(U)
X = hcat(X)

opti.minimize(sumsqr(U)+sumsqr(X))

opti.solver('ipopt')
sol = opti.solve()

Usol = sol.value(U) # should be [-2.7038;-0.5430;0.2613;0.5840]
print(Usol)

opti_MS = opti
U_MS = U
sol_MS = sol

print('MS Hessian')
hessian(opti_MS.f,opti_MS.x)[0].sparsity().spy()

print('MS jacobian')
jacobian(opti_MS.g,opti_MS.x).sparsity().spy()

print('SS Hessian')
hessian(opti_SS.f,opti_SS.x)[0].sparsity().spy()

print('SS jacobian')
jacobian(opti_SS.g,opti_SS.x).sparsity().spy()

# Exact same convergence: order will only slightly affect the factorization

## 1.5

# Multiple shooting is much sparser and therefore scales better for long time horizons

## 1.6


print(sol_MS.value(jacobian(opti_MS.g,opti_MS.x),opti_MS.initial()).toarray())
print(sol_SS.value(jacobian(opti_SS.g,opti_SS.x),opti_SS.initial()).toarray())

# Increasing N will increase further increase manitudes

# A slight change in u0 will affect x1 a little bit, x2 a bit more, x3 a huge amount, etc ...
# In other words, the effect of u0 blows up over time 

## 1.7
lag_MS = opti_MS.f+opti_MS.lam_g.T @ opti_MS.g

G_MS = jacobian(opti_MS.g,opti_MS.x)
KKT_MS = blockcat([[hessian(lag_MS,opti_MS.x)[0],G_MS.T],[G_MS,DM(opti_MS.ng,opti_MS.ng)]])

lag_SS = opti_SS.f+opti_SS.lam_g.T @ opti_SS.g

G_SS = jacobian(opti_SS.g,opti_SS.x)
KKT_SS = blockcat([[hessian(lag_SS,opti_SS.x)[0],G_SS.T],[G_SS,DM(opti_SS.ng,opti_SS.ng)]])

print(np.linalg.cond(sol_MS.value(KKT_MS,opti_MS.initial()).toarray())) # 16.5
print(np.linalg.cond(sol_SS.value(KKT_SS,opti_SS.initial()).toarray())) # 2.805e+20 

# The range of large to small numbers in the constraint Jacobian makes for an ill conditioned problem

## 1.8

# Second order
# Fourth order

# Linearization will be increasingly non-exact as order increases

# 1.9

# For MS: directly apply to state decision variables
# For SS: not really usable

# 1.10

N = 4
n = 3
m = 2

A = blockcat([[1,0.1,0.2],[2,0.3,0.4],[6,1,3]])
B = blockcat([[1,0],[0,1],[2,1]])

F = lambda x,u: A @ x+B @ u

opti = casadi.Opti()

X = opti.variable(n,N+1)
U = opti.variable(m,N)

opti.subject_to(X[:,0]==vertcat(1,2,3))
for k in range(N):
  opti.subject_to(F(X[:,k],U[:,k])==X[:,k+1])

opti.subject_to(X[:,-1]==0)

opti.minimize(sumsqr(U)+sumsqr(X))

opti.solver('ipopt')
sol = opti.solve()

print(sol.value(U[:,0])) # Expected [-4.2894;-4.1930]

# One iteration since its a QP. Same would hold for SS.

# 1.11

print('MS Hessian')
hessian(opti.f,opti.x)[0].sparsity().spy()

print('MS jacobian')
jacobian(opti.g,opti.x).sparsity().spy()

#  Stable dynamics -> no danger from states blowing up like with the x^2+u system
# No path constraints -> no dense triangular constraint Jacobian sparsity for SS
# dF/dx is 100-by-100 dense

# MS would create a constraint Jacobian with 100 million entries; might melt your computer.
# SS sounds like a good idea in this case, but not with exact Hessian!