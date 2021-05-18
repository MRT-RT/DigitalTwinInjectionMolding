# -*- coding: utf-8 -*-


# Import modules
import numpy as np

# Import sphere function as objective function
from pyswarms.utils.functions.single_obj import sphere as f

# Import backend modules
import pyswarms.backend as P
from pyswarms.backend.topology import Star

from pyswarms.single import GlobalBestPSO

from pyswarms.discrete.binary import BinaryPSO

from DiscreteBoundedPSO import DiscreteBoundedPSO
# my_topology = Star() # The Topology Class

# my_swarm = P.create_swarm(n_particles=50, dimensions=2, options=my_options) # The Swarm Class

# print('The following are the attributes of our swarm: {}'.format(my_swarm.__dict__.keys()))
options = {'c1': 0.6, 'c2': 0.3, 'w': 0.4, 'k':10, 'p':1} # arbitrarily set

param_bounds = {'dim_hidden':np.array([2,10])}


lb = np.array([2,2])
ub = np.array([2,10])

test = DiscreteBoundedPSO(n_particles=10, dimensions_discrete=2, options=my_options, bounds =(lb,ub))
test.optimize(f, iters=100)




# optimizer = GlobalBestPSO(n_particles=50, dimensions=2, options=my_options,bounds =(lb,ub)) # Reuse our previous options
# optimizer.optimize(f, iters=100)





# iterations = 100 # Set 100 iterations
# for i in range(iterations):
#     # Part 1: Update personal best
#     my_swarm.current_cost = f(my_swarm.position) # Compute current cost
#     my_swarm.pbest_cost = f(my_swarm.pbest_pos)  # Compute personal best pos
#     my_swarm.pbest_pos, my_swarm.pbest_cost = P.compute_pbest(my_swarm) # Update and store

#     # Part 2: Update global best
#     # Note that gbest computation is dependent on your topology
#     if np.min(my_swarm.pbest_cost) < my_swarm.best_cost:
#         my_swarm.best_pos, my_swarm.best_cost = my_topology.compute_gbest(my_swarm)

#     # Let's print our output
#     if i%20==0:
#         print('Iteration: {} | my_swarm.best_cost: {:.4f}'.format(i+1, my_swarm.best_cost))

#     # Part 3: Update position and velocity matrices
#     # Note that position and velocity updates are dependent on your topology
#     my_swarm.velocity = my_topology.compute_velocity(my_swarm)
#     my_swarm.position = my_topology.compute_position(my_swarm)

# print('The best cost found by our swarm is: {:.4f}'.format(my_swarm.best_cost))
# print('The best position found by our swarm is: {}'.format(my_swarm.best_pos))