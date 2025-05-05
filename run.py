from woa import whale_optimization
from problem import sphere
from visual import animate_woa_contour


# fitness_func: evaluation function;
# dim: dimension of the solution. For example: for the solution (x, y), we need two parameters. so "dim = 2"
# bounds: scope of each dimension of the solution
# num_agents: initial number of agents. population of whales
# max_iter: iterations to optimize
# integer_dims：If values of some dims should be integer
best, score= whale_optimization(fitness_func=sphere,dim=2,bounds=[(-5,5),(-5, 5)],num_agents= 5,max_iter= 100, integer_dims= [0,1])
# low: 11 - 40
# high: 1-10
# animate_woa_contour(trajectory, sphere, bounds=[(-5, 5), (-5, 5)])

print(f"Best solution: {best.tolist()}")
print(f"Best score: {score}")