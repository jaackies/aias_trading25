from algos.base import BasePopAlgo
import numpy as np


class PSO(BasePopAlgo):
    """
    Particle Swarm Optimisation (PSO) algorithm.
    """

    def _algo_init(self):
        """
        Initialize the PSO algorithm with the given parameters.
        :param num_particles: Number of particles in the swarm.
        """
        self.w = 0.5  # Inertia weight
        self.c1 = 1.5  # Cognitive parameter
        self.c2 = 1.5  # Social parameter

        self.pop_velocity = np.array(
            [
                self.rand_gen.uniform(
                    -np.abs(high - low), np.abs(high - low), self.pop_size
                )
                for low, high in self.bounds
            ]
        ).T
        self.pop_best_params = self.pop.copy()
        self.pop_best_fitness = self.pop_fitness.copy()

    def _algo_iter(self, _):
        for i in range(self.pop_size):  # each particle
            # each param
            for ii in range(len(self.bounds)):
                # Update velocity
                r1, r2 = self.rand_gen.random(2)
                self.pop_velocity[i][ii] = (
                    self.w * self.pop_velocity[i][ii]
                    + self.c1 * r1 * (self.pop_best_params[i][ii] - self.pop[i][ii])
                    + self.c2 * r2 * (self.best_params[ii] - self.pop[i][ii])
                )

                # Update position
                self.pop[i][ii] += self.pop_velocity[i][ii]

                # Bound the position
                self.pop[i][ii] = max(
                    self.bounds[ii][0],
                    min(self.bounds[ii][1], self.pop[i][ii]),
                )
                if ii in self.integer_dims:
                    self.pop[i][ii] = round(self.pop[i][ii])

            # Evaluate fitness and update global best
            self.pop_fitness[i] = self.eval(self.pop[i])
            if self.pop_fitness[i] > self.best_fitness:
                self.best_params = self.pop[i]
                self.best_fitness = self.pop_fitness[i]
            # Update the best parameters for each particle
            if self.pop_fitness[i] > self.pop_best_fitness[i]:
                self.pop_best_params[i] = self.pop[i]
                self.pop_best_fitness[i] = self.pop_fitness[i]
