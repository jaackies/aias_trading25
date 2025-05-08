import numpy as np

from algos.base import BaseAlgo


class HC(BaseAlgo):
    def _algo_init(self, _):
        self.high_window = None
        self.low_window = None
        self.alpha = 0
        self.new_high_frequency_window = high_window
        self.new_low_frequency_window = low_window
        self.new_alpha = 0

        self.high_window = [
            (
                self.rand_gen.integers
                if i in self.integer_dims
                else self.rand_gen.uniform
            )(low, high)
            for i, (low, high) in enumerate(self.bounds)
        ]

    def _algo_iter(self, _):
        for i in range(self.num_agents):
            r, p = self.rand_gen.random(2)
            A = 2 * self.a * r - self.a  # Update A
            C = 2 * r  # Update C

            if p < 0.5:
                if np.linalg.norm(A) >= 1:  # Exploration phase
                    X_rand = self.X[self.rand_gen.integers(0, self.num_agents)]
                    D = np.abs(C * X_rand - self.X[i])
                    self.X[i] = X_rand - A * D
                else:  # Exploitation phase - Shrinking encircling mechanism
                    D = np.abs(C * self.best_params - self.X[i])
                    self.X[i] = self.best_params - A * D
            else:  # if p >= 0.5；Exploitation phase - Spiral updating position
                D = np.abs(self.best_params - self.X[i])
                b = 1
                l = self.rand_gen.uniform(-1, 1, self.dim)
                self.X[i] = D * np.exp(b * l) * np.cos(2 * np.pi * l) + self.best_params

            # Check if any search agent goes beyond the search space and amend it
            for d in range(self.dim):
                self.X[i][d] = np.clip(
                    self.X[i][d], self.bounds[d][0], self.bounds[d][1]
                )
                if d in self.integer_dims:
                    self.X[i][d] = int(round(self.X[i][d]))

            # Calculate the fitness of each search agent
            score = self._eval_and_update(self.X[i])
            # print(f"\tWhale: {i}, Score: {score}, Solution: {self.X[i]}")

        self.a -= self.a_dec_v
