
from sys import path
# path.append(r"C:\Users\LocalAdmin\Documents\casadi-windows-py38-v3.5.5-64bit")
from casadi import *

# Lotka-Volerra problem from mintoc.de

N = 40 # number of control intervals

# Initial condition for x
x0 = 1.0

# Bounds on x
lbx = [0.]
ubx = [2.]


# Declare model variables
x = SX.sym('x', 1)
t = SX.sym('x', 1)

# Model equations
x_new = 0.9*x

# Objective term
L = (x[0] - 1)**2

# Formulate discrete time dynamics
F = Function('F', [x], [x_new])


# if False:
#    # CVODES from the SUNDIALS suite
#    dae = {'x':x, 'p':u, 'ode':xdot, 'quad':L}
#    opts = {'tf':T/N}
#    F = integrator('F', 'cvodes', dae, opts)
# else:
#    # Fixed step Runge-Kutta 4 integrator
#    M = 4 # RK4 steps per interval
#    DT = T/N/M
#    f = Function('f', [x, u], [xdot, L])
#    X0 = MX.sym('X0', 2)
#    U = MX.sym('U')
#    X = X0
#    Q = 0
#    for j in range(M):
#        k1, k1_q = f(X, U)
#        k2, k2_q = f(X + DT/2 * k1, U)
#        k3, k3_q = f(X + DT/2 * k2, U)
#        k4, k4_q = f(X + DT * k3, U)
#        X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
#        Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
#    F = Function('F', [X0], [X],['x0'],['xf'])


# Get a feasible trajectory as an initial guess
xk = DM(x0)
x_start = [xk]
for k in range(N):
    xk = F(xk)
    x_start += [xk]




# Start with an empty NLP
w=[]
w0 = []
lbw = []
ubw = []
discrete = []
g=[]
lbg = []
ubg = []

# Time Optimal
k = MX.sym('k', 1)
k_start = 15
w += [k]
lbw += [10]
ubw += [20]
w0 += [k_start]
discrete += [True]

# "Lift" initial conditions
X0 = MX.sym('X0', 1)
w += [X0]
lbw += [x0]
ubw += [x0]
w0 += [x_start[0]]
discrete += [False]

# Formulate the NLP
Xk = X0
for i in range(N):

    # Integrate till the end of the interval
    Fk = F(Xk)
    Xk_end = Fk

    # New NLP variable for state at end of interval
    Xk = MX.sym('X_' + str(i+1), 1)
    w   += [Xk]
    lbw += lbx
    ubw += ubx
    w0  += [x_start[i+1]]
    discrete += [False]

    # Add equality constraint
    g   += [Xk_end-Xk]
    lbg += [0]
    ubg += [0]

J = k        

g   += [w[k]-1*0.9**15]
lbg += [0]
ubg += [0]

# Concatenate decision variables and constraint terms
w = vertcat(*w)
g = vertcat(*g)

# Create an NLP solver
nlp_prob = {'f': J, 'x': w, 'g': g}
nlp_solver = nlpsol('nlp_solver', 'bonmin', nlp_prob, {"discrete": discrete});
# nlp_solver = nlpsol('nlp_solver', 'knitro', nlp_prob, {"discrete": discrete});
# nlp_solver = nlpsol('nlp_solver', 'ipopt', nlp_prob); # Solve relaxed problem

# Plot the solution
# tgrid = [T/N*k for k in range(N+1)]
# import matplotlib.pyplot as plt
# plt.figure(1)
# plt.clf()
# def plot_sol(w_opt):
#     w_opt = w_opt.full().flatten()
#     x0_opt = w_opt[0::3]
#     x1_opt = w_opt[1::3]
#     u_opt = w_opt[2::3]
#     plt.plot(tgrid, x0_opt, '--')
#     plt.plot(tgrid, x1_opt, '-')
#     plt.step(tgrid, vertcat(DM.nan(1), u_opt), '-.')
#     plt.xlabel('t')
#     plt.legend(['x0','x1','u'])
#     plt.grid(True)

# Solve the NLP
sol = nlp_solver(x0=vertcat(*w0), lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

print(nlp_solver.stats())

w1_opt = sol['x']
lam_w_opt = sol['lam_x']
lam_g_opt = sol['lam_g']
plot_sol(w1_opt)

plt.show()
