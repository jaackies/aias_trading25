from woa import whale_optimization
from problem import bot_fitness_func

# fitness_func: evaluation function;
# dim: dimension of the solution. For example: for the solution (x, y), we need two parameters. so "dim = 2"
# bounds: scope of each dimension of the solution
# num_agents: initial number of agents. population of whales
# max_iter: iterations to optimize
# integer_dims：If values of some dims should be integer
# bot_type: "sma", "smaema"， "complex"
best, score= whale_optimization(fitness_func=bot_fitness_func, bot_type="sma", dim=2, bounds=[(5,50),(20,100)], num_agents= 10, max_iter= 100, integer_dims= [0,1])
# low: 11 - 40， 51,300
# high: 1-10 5,50

print(f"Best solution: {best.tolist()}")
print(f"Best score: {score}")