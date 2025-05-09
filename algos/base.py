import numpy as np


class BaseAlgo:
    """
    Base class for all optimisation algorithms.
    """

    def __init__(
        self,
        eval_func,
        *,
        bounds,
        integer_dims: frozenset = frozenset(),
        seed=None,
        max_iter=1000,
    ):
        """

        Initializes the BaseAlgo with an evaluation function (which returns a fitness value where bigger is better) and bounds for the parameters (solution).

        Lower bound inclusive, upper bound exclusive.
        """
        self.max_iter = max_iter
        self.__eval_func = eval_func  # Function takes in an array of parameters and returns a fitness value
        self.bounds = bounds
        self.integer_dims = integer_dims

        self.best_params = None
        self.best_fitness = -np.inf

        self.rand_gen = np.random.default_rng(seed)

        self.__fitness_at_evals = [0]

    @property
    def dim(self):
        return len(self.bounds)

    @property
    def name(self):
        return self.__class__.__name__

    def eval(self, candidate):
        """
        Evaluates the candidate solution and returns the fitness value.
        """
        self.__fitness_at_evals.append(self.best_fitness)
        return self.__eval_func(candidate)

    def _algo_init(self, **kwargs):
        """
        This method should be overridden by subclasses to define the initialization of the algorithm.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def _algo_iter(self, iter_num):
        """
        This method should be overridden by subclasses to define what happens in each iteration of the algorithm.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def optimise(self, **kwargs):
        self._algo_init(self.max_iter, **kwargs)
        for i in range(self.max_iter):
            self._algo_iter(i)

    def plot(self, title="Fitness Over Evaluations"):
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(y=self.__fitness_at_evals, mode="lines", name="Fitness")
        )
        fig.update_layout(
            title=title,
            xaxis_title="Evaluations",
            yaxis_title="Fitness",
            template="plotly_white",
        )
        fig.show()


class BasePopAlgo(BaseAlgo):
    """
    Base class for all population-based optimisation algorithms.
    """

    def __init__(
        self,
        eval_func,
        bounds,
        integer_dims=frozenset(),
        seed=None,
        pop_size=10,
        max_iter=100,
    ):
        super().__init__(eval_func, bounds, integer_dims, seed, max_iter)
        self.pop = np.array(
            (
                self.rand_gen.integers
                if i in self.integer_dims
                else self.rand_gen.uniform
            )(low, high, pop_size)
            for i, (low, high) in enumerate(self.bounds)
        ).T
        self.pop_fitness = np.array([self.eval(p) for p in self.pop])
        self.best_params = self.pop[np.argmax(self.pop_fitness)]
        self.best_fitness = np.max(self.pop_fitness)
