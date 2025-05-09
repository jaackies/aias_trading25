import numpy as np


class BaseAlgo:
    """
    Base class for all optimisation algorithms.

    NOTE: these are all maximising optimisation algorithms.
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
        self.rand_gen = np.random.default_rng(seed)

        self.historical_best_params = []  # SHOULDN'T BE TOUCHED BY SUBCLASSES

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
        self.fitness_over_evals.append(self.best_fitness)
        return self.__eval_func(candidate)

    def _algo_init(self):
        """
        This method should be overridden by subclasses to define the initialization of the algorithm.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def _algo_iter(self, iter_num):
        """
        This method should be overridden by subclasses to define what happens in each iteration of the algorithm.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def _prep(self):
        self.best_params = None
        self.best_fitness = -np.inf
        self.fitness_over_evals = [0]

    def optimise(self):
        self._prep()
        self._algo_init()
        for i in range(self.max_iter):
            self._algo_iter(i)
        self.historical_best_params.append(self.best_params)

    def historical_best_params_fitness(self, eval_fn):
        return [eval_fn(p) for p in self.historical_best_params]

    def plot(self):
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(y=self.fitness_over_evals, mode="lines", name="Fitness")
        )
        fig.update_layout(
            xaxis_title="Evaluations",
            yaxis_title="Fitness",
            template="plotly_white",
        )
        return fig


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
        self.pop_size = pop_size
        super().__init__(
            eval_func,
            bounds=bounds,
            integer_dims=integer_dims,
            seed=seed,
            max_iter=max_iter,
        )

    def _prep(self):
        super()._prep()
        self.pop = np.array(
            [
                (
                    self.rand_gen.integers
                    if i in self.integer_dims
                    else self.rand_gen.uniform
                )(low, high, self.pop_size)
                for i, (low, high) in enumerate(self.bounds)
            ]
        ).T
        self.pop_fitness_over_evals = []
        self.pop_fitness = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            self.pop_fitness[i] = self.eval(self.pop[i])
        self.best_params = self.pop[np.argmax(self.pop_fitness)]
        self.best_fitness = np.max(self.pop_fitness)

    def eval(self, candidate):
        self.pop_fitness_over_evals.append(self.pop_fitness.copy())
        return super().eval(candidate)

    def plot(self):
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(y=self.fitness_over_evals, mode="lines", name="Fitness")
        )
        self.pop_fitness_over_evals = np.array(self.pop_fitness_over_evals)
        for i in range(self.pop_size):
            fig.add_trace(
                go.Scatter(
                    y=self.pop_fitness_over_evals.T[i],
                    mode="markers",
                    marker=dict(size=1, opacity=0.75, color="orange"),
                    name=f"Agent {i}",
                    showlegend=False,
                )
            )
        fig.update_layout(
            xaxis_title="Evaluations",
            yaxis_title="Fitness",
            template="plotly_white",
        )
        return fig
