import numpy as np

from algos.base import BaseAlgo


class HC(BaseAlgo):  # steepest ascent hill climbing with replacement
    def _algo_init(self, _, no_of_tweaks=10):
        self._eval_and_update(
            [
                (
                    self.rand_gen.integers
                    if i in self.integer_dims
                    else self.rand_gen.uniform
                )(low, high)
                for i, (low, high) in enumerate(self.bounds)
            ]
        )

        self.S = self.best_params
        self.S_fitness = self.best_fitness

        self.no_of_tweaks = no_of_tweaks

    def _algo_iter(self, _):
        R = self.tweak(self.S)
        R_fitness = self._eval(R)
        for _ in range(self.no_of_tweaks):
            W = self.tweak(R)
            W_fitness = self._eval(W)
            if W_fitness > R_fitness:
                R = W
                R_fitness = W_fitness
        if R_fitness > self.S_fitness:
            self.S = R
            self.S_fitness = R_fitness
        if self.S_fitness > self.best_fitness:
            self.best_params = self.S
            self.best_fitness = self.S_fitness

    def tweak(self, v: list, r=None, p=1):  # bounded uniform convolution
        if r is None:
            r = 0.1 * (self.bounds[0][1] - self.bounds[0][0])

        mod_params = v.copy()

        for i in range(len(mod_params)):
            if p >= self.rand_gen.uniform(0, 1):
                while True:
                    n = self.rand_gen.uniform(-r, r)
                    lbound, hbound = self.bounds[i]
                    potential = mod_params[i] + n
                    if i in self.integer_dims:
                        potential = int(potential)
                    if lbound <= potential <= hbound:
                        mod_params[i] = potential
                        break

        return mod_params
