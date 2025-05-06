import numpy as np

from algos.base import BaseAlgo


class WOA(BaseAlgo):
    """
    Whale Optimization Algorithm (WOA) for optimization problems.
    """

    def _algo_init(self, max_iter, *, num_agents=30):
        """
        Initialize the WOA algorithm with the given parameters.
        :param max_iter: Maximum number of iterations.
        :param num_agents: Number of agents (whales).
        :param integer_dims: List of indices for dimensions that should be treated as integers.
        """
        self.num_agents = num_agents

        self.a = 2
        self.a_dec_v = 2 / max_iter
        # Initialize the whales positions (which are initial solutions)
        self.X = np.array(
            [np.random.uniform(low, high, num_agents) for (low, high) in self.bounds]
        ).T
        for i in range(num_agents):
            self._eval_and_update(self.X[i])

    def _algo_iter(self, _):
        for i in range(self.num_agents):
            r = np.random.rand(self.dim)
            A = 2 * self.a * r - self.a  # Update A
            C = 2 * r  # Update C
            p = np.random.rand()

            if p < 0.5:
                if np.linalg.norm(A) >= 1:  # Exploration phase
                    X_rand = self.X[np.random.randint(0, self.num_agents)]
                    D = np.abs(C * X_rand - self.X[i])
                    self.X[i] = X_rand - A * D
                else:  # Exploitation phase - Shrinking encircling mechanism
                    D = np.abs(C * self.best_params - self.X[i])
                    self.X[i] = self.best_params - A * D
            else:  # if p >= 0.5；Exploitation phase - Spiral updating position
                D = np.abs(self.best_params - self.X[i])
                b = 1
                l = np.random.uniform(-1, 1, self.dim)
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
