import numpy as np


class BaseAlgo:
    """
    Base class for all optimisation algorithms.
    """

    def __init__(
        self, eval_func, bounds, integer_dims: frozenset = frozenset(), seed=None
    ):
        """

        Initializes the BaseAlgo with an evaluation function (which returns a fitness value where bigger is better) and bounds for the parameters (solution).

        Lower bound inclusive, upper bound exclusive.
        """
        self.__eval_func = eval_func  # Function takes in an array of parameters and returns a fitness value
        self.bounds = bounds
        self.integer_dims = integer_dims

        self.best_params = None
        self.best_fitness = -np.inf

        self.rand_gen = np.random.default_rng(seed)

    @property
    def dim(self):
        return len(self.bounds)

    @property
    def name(self):
        return self.__class__.__name__

    def _eval_and_update(self, candidate):
        """
        Updates the best solution found so far if the candidate solution is better. Returns the fitness value of the candidate solution either way.
        """
        fitness = self.__eval_func(candidate)
        if fitness > self.best_fitness:
            self.best_params = candidate.copy()
            self.best_fitness = fitness
        return fitness

    def _algo_init(self, max_iter, **kwargs):
        """
        This method should be overridden by subclasses to define the initialization of the algorithm.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def _algo_iter(self, iter_num):
        """
        This method should be overridden by subclasses to define what happens in each iteration of the algorithm.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def optimise(self, max_iter=50, **kwargs):
        self._algo_init(max_iter, **kwargs)
        for iter_num in range(max_iter):
            # print(
            #     f"Iteration {iter_num + 1}/{max_iter}, Best Solution: {self.best}, Best Fitness: {self.best_fitness}"
            # )
            self._algo_iter(iter_num)
