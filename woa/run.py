from woa import whale_optimization
from problem import sphere

best, score = whale_optimization(fitness_func=sphere,dim=2,bounds=(-100, 100),num_agents=2,max_iter=5)
# fitness_func: evaluation function;
# dim: dimension of the solution. For example: for the solution (x, y), we need two parameters. so "dim = 2"
# bounds: scope of each dimension of the solution
# num_agents: initial number of agents. population of whales
# max_iter: iterations to optimize

print("Best solution:", best)
print("Best score:", score)