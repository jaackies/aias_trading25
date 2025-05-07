from woa import whale_optimization
from bot_training import bot_training

# fitness_func: evaluation function;
# dim: dimension of the solution. For example: for the solution (x, y), we need two parameters. so "dim = 2"
# bounds: scope of each dimension of the solution
# num_agents: initial number of agents. population of whales
# max_iter: iterations to optimize
# integer_dims：If values of some dims should be integer
# bot_type: "sma", "smaema"， "complex"
# bot_type = "complex" bounds bounds=[(1, 10), (1, 10), (1, 10), # w1, w2, w3 (2, 50), (2, 50), (2, 50), # d1, d2, d3 (0, 1), # sf # -- Low part --(1, 10), (1, 10), (1, 10),(2, 90), (2, 90), (2, 90),(0, 1)]
# best, score= whale_optimization(fitness_func=bot_training, bot_type="sma", dim=2, bounds=[(1,10), (11,40)], num_agents= 10, max_iter= 100, integer_dims= [0,1])
# best, score= whale_optimization(fitness_func=bot_training, bot_type="smaema", dim=3, bounds=[(1,10), (11,40), (0,1)], num_agents= 10, max_iter= 100, integer_dims= [0,1])
best, score= whale_optimization(fitness_func=bot_training, bot_type="complex", dim=14, bounds=[(1, 10), (1, 10), (1, 10), (2, 50), (2, 50), (2, 50), (0, 1), (1, 10), (1, 10), (1, 10),(2, 90), (2, 90), (2, 90),(0, 1)], num_agents= 10, max_iter= 10, integer_dims= [3,4,5,10,11,12])
# low: (11, 40)， (51,100)
# high: (1, 10), (5, 50)

print(f"Best solution: {best}")
print(f"Best score: {score}")