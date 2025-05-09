import numpy as np

from algos.base import BaseAlgo


class HC(BaseAlgo):  # steepest ascent hill climbing with replacement
    def _algo_init(self, no_of_additional_tweaks=0):
        self.best_params = [
            (
                self.rand_gen.integers
                if i in self.integer_dims
                else self.rand_gen.uniform
            )(low, high)
            for i, (low, high) in enumerate(self.bounds)
        ]
        self.best_fitness = self.eval(self.best_params)

        self.S = self.best_params
        self.S_fitness = self.best_fitness

        self.no_of_additional_tweaks = no_of_additional_tweaks

    def _algo_iter(self, _):
        R = self.tweak(self.S)
        R_fitness = self.eval(R)
        for _ in range(self.no_of_additional_tweaks):
            W = self.tweak(R)
            W_fitness = self.eval(W)
            if W_fitness > R_fitness:
                R = W
                R_fitness = W_fitness
        if R_fitness > self.S_fitness:
            self.S = R
            self.S_fitness = R_fitness
        if self.S_fitness > self.best_fitness:
            self.best_params = self.S
            self.best_fitness = self.S_fitness

    def tweak(self, v: list, r_percent=0.1, p=1):  # bounded uniform convolution
        r = r_percent * (self.bounds[0][1] - self.bounds[0][0])

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
