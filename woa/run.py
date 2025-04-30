from woa import whale_optimization
from problem import sphere


# fitness_func: evaluation function;
# dim: dimension of the solution. For example: for the solution (x, y), we need two parameters. so "dim = 2"
# bounds: scope of each dimension of the solution
# num_agents: initial number of agents. population of whales
# max_iter: iterations to optimize
# integer_dims：If values of some dims should be integer
best, score = whale_optimization(fitness_func=sphere,dim=3,bounds=[(-5,5),(3, 5),(1, 3)],num_agents=2,max_iter=5)

print(f"Best solution: {best.tolist()}")
print(f"Best score: {score}")