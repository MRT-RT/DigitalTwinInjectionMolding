# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 16:33:48 2021

@author: LocalAdmin
"""

# Import standard library
import logging

# Import modules
import numpy as np
import multiprocessing as mp

from collections import deque


from pyswarms.discrete.binary import BinaryPSO
from pyswarms.backend.operators import compute_pbest, compute_objective_function
from pyswarms.backend.topology import Ring
from pyswarms.backend.handlers import BoundaryHandler, VelocityHandler
from pyswarms.base import DiscreteSwarmOptimizer
from pyswarms.utils.reporter import Reporter


class DiscreteBoundedPSO(BinaryPSO):
    """
    This class is based on the Binary PSO class. It extends the BinaryPSO class
    by a function which allows the conversion of discrete optimization variables
    into binary variables, so that discrete optimization problems can be solved 
    """
    def __init__(
        self,
        n_particles,
        dimensions_discrete,
        options,
        bounds,
        bh_strategy="periodic",
        init_pos=None,
        velocity_clamp=None,
        vh_strategy="unmodified",
        ftol=-np.inf,
        ftol_iter=1,
    ):
        """Initialize the swarm

        Attributes
        ----------
        n_particles : int
            number of particles in the swarm.
        dimensions_discrete : int
            number of discrete dimensions of the search space.
        options : dict with keys :code:`{'c1', 'c2', 'w', 'k', 'p'}`
            a dictionary containing the parameters for the specific
            optimization technique
                * c1 : float
                    cognitive parameter
                * c2 : float
                    social parameter
                * w : float
                    inertia parameter
                * k : int
                    number of neighbors to be considered. Must be a
                    positive integer less than :code:`n_particles`
                * p: int {1,2}
                    the Minkowski p-norm to use. 1 is the
                    sum-of-absolute values (or L1 distance) while 2 is
                    the Euclidean (or L2) distance.
        bounds : tuple of numpy.ndarray
            a tuple of size 2 where the first entry is the minimum bound while
            the second entry is the maximum bound. Each array must be of shape
            :code:`(dimensions,)`.
        init_pos : numpy.ndarray, optional
            option to explicitly set the particles' initial positions. Set to
            :code:`None` if you wish to generate the particles randomly.
        velocity_clamp : tuple, optional
            a tuple of size 2 where the first entry is the minimum velocity
            and the second entry is the maximum velocity. It
            sets the limits for velocity clamping.
        vh_strategy : String
            a strategy for the handling of the velocity of out-of-bounds particles.
            Only the "unmodified" and the "adjust" strategies are allowed.
        ftol : float
            relative error in objective_func(best_pos) acceptable for
            convergence
        ftol_iter : int
            number of iterations over which the relative error in
            objective_func(best_pos) is acceptable for convergence.
            Default is :code:`1`
        """
        # Initialize logger
        self.rep = Reporter(logger=logging.getLogger(__name__))
        # Assign k-neighbors and p-value as attributes
        self.k, self.p = options["k"], options["p"]
        
        self.dimensions_discrete = dimensions_discrete
        
        self.bits,self.bounds = self.discretePSO_to_binaryPSO(
            dimensions_discrete,bounds)
        
        
        # Initialize parent class
        super(BinaryPSO, self).__init__(
            n_particles=n_particles,
            dimensions=sum(self.bits),
            binary=True,
            options=options,
            init_pos=init_pos,
            velocity_clamp=velocity_clamp,
            ftol=ftol,
            ftol_iter=ftol_iter,
        )
        # self.bounds = bounds
        # Initialize the resettable attributes
        self.reset()
        # Initialize the topology
        self.top = Ring(static=False)
        self.vh = VelocityHandler(strategy=vh_strategy)
        self.bh = BoundaryHandler(strategy=bh_strategy)
        self.name = __name__
        
        
        
    def optimize(
        self, objective_func, iters, n_processes=None, verbose=True, **kwargs
        ):
        """Optimize the swarm for a number of iterations

        Performs the optimization to evaluate the objective
        function :code:`f` for a number of iterations :code:`iter.`

        Parameters
        ----------
        objective_func : function
            objective function to be evaluated
        iters : int
            number of iterations
        n_processes : int, optional
            number of processes to use for parallel particle evaluation
            Defaut is None with no parallelization.
        verbose : bool
            enable or disable the logs and progress bar (default: True = enable logs)
        kwargs : dict
            arguments for objective function

        Returns
        -------
        tuple
            the local best cost and the local best position among the
            swarm.
        """
        # Apply verbosity
        if verbose:
            log_level = logging.INFO
        else:
            log_level = logging.NOTSET

        self.rep.log("Obj. func. args: {}".format(kwargs), lvl=logging.DEBUG)
        self.rep.log(
            "Optimize for {} iters with {}".format(iters, self.options),
            lvl=log_level,
        )
        # Populate memory of the handlers
        self.bh.memory = self.swarm.position
        self.vh.memory = self.swarm.position

        # Setup Pool of processes for parallel evaluation
        pool = None if n_processes is None else mp.Pool(n_processes)

        self.swarm.pbest_cost = np.full(self.swarm_size[0], np.inf)
        ftol_history = deque(maxlen=self.ftol_iter)
        for i in self.rep.pbar(iters, self.name) if verbose else range(iters):
            
            # Compute cost for current position and personal best
            
            ''' Binary swarm postitions need to be transformed to discrete swarm
                postitions first, because the objective function expects discrete
                values (only positions are transformed!), original binary
                position is saved in binary_swarm_position'''
            binary_swarm_position = self.BinarySwarmPositions_to_DiscreteSwarmPositions()
                   
            # Evaluate Cost Function
            self.swarm.current_cost = compute_objective_function(
                self.swarm, objective_func, pool, **kwargs
            )
            
            ''' Transform discrete swarm positions back to binary positions, 
            because the PSO works on binary particles'''
            self.swarm.position = binary_swarm_position
            
            self.swarm.pbest_pos, self.swarm.pbest_cost = compute_pbest(
                self.swarm
            )
            best_cost_yet_found = np.min(self.swarm.best_cost)
            # Update gbest from neighborhood
            self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(
                self.swarm, p=self.p, k=self.k
            )
            if verbose:
                # Print to console
                self.rep.hook(best_cost=self.swarm.best_cost)
            # Save to history
            hist = self.ToHistory(
                best_cost=self.swarm.best_cost,
                mean_pbest_cost=np.mean(self.swarm.pbest_cost),
                mean_neighbor_cost=np.mean(self.swarm.best_cost),
                position=self.swarm.position,
                velocity=self.swarm.velocity,
            )
            self._populate_history(hist)
            # Verify stop criteria based on the relative acceptable cost ftol
            relative_measure = self.ftol * (1 + np.abs(best_cost_yet_found))
            delta = (
                np.abs(self.swarm.best_cost - best_cost_yet_found)
                < relative_measure
            )
            if i < self.ftol_iter:
                ftol_history.append(delta)
            else:
                ftol_history.append(delta)
                if all(ftol_history):
                    break
            # Perform position velocity update
            self.swarm.velocity = self.top.compute_velocity(
                self.swarm, self.velocity_clamp, self.vh
            )
            self.swarm.position = self._compute_position(self.swarm)
            
        # Obtain the final best_cost and the final best_position
        final_best_cost = self.swarm.best_cost.copy()
        final_best_pos = self.swarm.pbest_pos[
            self.swarm.pbest_cost.argmin()
        ].copy()
        self.rep.log(
            "Optimization finished | best cost: {}, best pos: {}".format(
                final_best_cost, final_best_pos
            ),
            lvl=log_level,
        )
        # Close Pool of Processes
        if n_processes is not None:
            pool.close()

        return (final_best_cost, final_best_pos)
    
    def discretePSO_to_binaryPSO(self,dimensions_discrete,bounds):
        """
        Translate a discrete PSO-problem into a binary PSO-problem by
        calculating the number of bits necessary to represent the discrete
        optimization problem with "dimensions_discrete" number of discrete
        variables as a binary optimization problem. The bounds are encoded in 
        the binary representation and might be tightened.
        
        Parameters
        ----------  
        dimensions_discrete: integer
            dimension of the discrete search space.
        bounds : tuple of numpy.ndarray
            a tuple of size 2 where the first entry is the minimum bound while
            the second entry is the maximum bound. Each array must be of shape
            :code:`(dimensions,)`.
        """
        
        bits = []
        
        for n in range(0,dimensions_discrete):
            
            # Number of bits required rounding down!
            bits.append(int(np.log10(bounds[1][n]-bounds[0][n]+1) / np.log10(2)))
        
            # Adjust upper bound accordingly
            bounds[1][n] = bounds[0][n] + 2**bits[n]-1
        
        return bits, bounds

    def BinarySwarmPositions_to_DiscreteSwarmPositions(self):
        """
        Converts binary self.swarm.position to discrete values. Returns the 
        original binary position, so that it can be used to restore 
        self.swarm.position to the original binary values.
        """
        
        
        binary_position = self.swarm.position
        discrete_position = np.zeros((self.n_particles,self.dimensions_discrete))
        
        cum_sum = 0
        
        for i in range(0,self.dimensions_discrete):
            
            bit = self.bits[i]
            lb = self.bounds[0][i]
            
            discrete_position[:,[i]] = lb + \
            self.bool2int(binary_position[:,cum_sum:cum_sum+bit])
            
            cum_sum = cum_sum + bit
        
        
        # Set swarm position to discrete integer values
        self.swarm.position = discrete_position.astype(int)
        
        return binary_position    
                    
    def bool2int(self,x):
        """
        Converts a binary variable represented by an array x (row vector) into
        an integer value
        """
        
        x_int = np.zeros((x.shape[0],1))
        
        for row in range(0,x.shape[0]):
            row_int = 0
            
            for i,j in enumerate(x[row,:]):
                row_int += j<<i
            
            x_int[row] = row_int
        
        return x_int          
    
        
     
        
        
        
        
        