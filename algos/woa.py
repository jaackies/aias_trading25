import numpy as np

from algos.base import BasePopAlgo


class WOA(BasePopAlgo):
    """
    Whale Optimization Algorithm (WOA) for optimization problems.
    """

    def _algo_init(self):
        """
        Initialize the WOA algorithm with the given parameters.
        :param max_iter: Maximum number of iterations.
        :param num_agents: Number of agents (whales).
        :param integer_dims: List of indices for dimensions that should be treated as integers.
        """

        self.a = 2
        self.a_dec_v = 2 / self.max_iter

    def _algo_iter(self, _):
        for i in range(self.pop_size):
            r, p = self.rand_gen.random(self.dim), self.rand_gen.random()
            A = 2 * self.a * r - self.a  # Update A
            C = 2 * r  # Update C

            if p < 0.5:
                if np.linalg.norm(A) >= 1:  # Exploration phase
                    X_rand = self.pop[self.rand_gen.integers(0, self.pop_size)]
                    D = np.abs(C * X_rand - self.pop[i])
                    self.pop[i] = X_rand - A * D
                else:  # Exploitation phase - Shrinking encircling mechanism
                    D = np.abs(C * self.best_params - self.pop[i])
                    self.pop[i] = self.best_params - A * D
            else:  # if p >= 0.5；Exploitation phase - Spiral updating position
                D = np.abs(self.best_params - self.pop[i])
                b = 1
                l = self.rand_gen.uniform(-1, 1, self.dim)
                self.pop[i] = (
                    D * np.exp(b * l) * np.cos(2 * np.pi * l) + self.best_params
                )

            # Check if any search agent goes beyond the search space and amend it
            for d in range(self.dim):
                self.pop[i][d] = np.clip(self.pop[i][d], *self.bounds[d])
                if d in self.integer_dims:
                    self.pop[i][d] = int(round(self.pop[i][d]))

            # Calculate the fitness of each search agent
            self.pop_fitness[i] = self.eval(self.pop[i])
            if self.pop_fitness[i] > self.best_fitness:
                self.best_params = self.pop[i].copy()
                self.best_fitness = self.pop_fitness[i]

        self.a -= self.a_dec_v
