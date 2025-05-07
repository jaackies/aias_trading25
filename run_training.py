from woa import whale_optimization
from bot_training import bot_training
from hill_climbing import hill_climbing

#Group A:
# High: (1, 10)
# Low: (11, 40)

#Group B:
# High:  (5, 50)
# Low: (51,100)

########### WOA ################
# best, score= whale_optimization(fitness_func=bot_training, bot_type="sma", dim=2, bounds=[(1,10), (11,40)], num_agents= 10, max_iter= 100, integer_dims= [0,1])
# best, score= whale_optimization(fitness_func=bot_training, bot_type="smaema", dim=3, bounds=[(1,10), (11,40), (0,1)], num_agents= 10, max_iter= 100, integer_dims= [0,1])
# best, score= whale_optimization(fitness_func=bot_training, bot_type="complex", dim=14, bounds=[(0, 10), (0, 10), (0, 10), (1, 10), (1, 10), (1, 10), (0, 1), (0, 10), (0, 10), (0, 10),(11, 40), (11, 40), (11, 40),(0, 1)], num_agents= 10, max_iter= 100, integer_dims= [3,4,5,10,11,12])
# print(f"Best solution: {best}")
# print(f"Best score: {score}")

########## Hill-Climbing ########
# sma_optimals, cash_result = hill_climbing('sma',[(1,10),(11,40)])
# samema_optimals, cash_result = hill_climbing('smaema', [(1,10),(11,40),(0,1)])
# print(samema_optimals) 
# print(cash_result)
complex_optimals, cash_result = hill_climbing('complex',[(0,10),(0,10),(0,10),(1,10),(1,10),(1,10),(0,1), (0,10),(0,10),(0,10),(11,40),(11,40),(11,40),(0,1)])
print(complex_optimals)
# print(cash_result)
# print(sma_optimals)
# print(cash_result)