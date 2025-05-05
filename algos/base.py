class BaseAlgo:
    """
    Base class for all optimisation algorithms.
    """

    def __init__(self, eval_func, bounds):
        self.eval_func = eval_func  # Function takes in an array of parameters and returns a fitness value
        self.bounds = bounds

    def optimise(self):
        """
        Optimise the function. Return the best parameters and their fitness value.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
