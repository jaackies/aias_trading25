from woa import whale_optimization
from bot_training import bot_training
from bot_testing import bot_testing
from hill_climbing import hill_climbing

results = {
    'WOA': [],
    'PSO': [],
    'HC': []
}

for _ in range(10):
    # WOA
    woa_params = whale_optimization(fitness_func=bot_training, bot_type="complex", dim=14, bounds=[(0, 10), (0, 10), (0, 10), (1, 10), (1, 10), (1, 10), (0, 1), (0, 10), (0, 10), (0, 10),(11, 40), (11, 40), (11, 40),(0, 1)], num_agents= 10, max_iter= 10, integer_dims= [3,4,5,10,11,12])
    cash = bot_testing("complex", woa_params[0])
    results['WOA'].append(cash)

    # # PSO
    # bso_params = run_bso()
    # cash = testing_bot(*bso_params)
    # results['BSO'].append(cash)

    # HC
    HC_params = hill_climbing('complex',[(0,10),(0,10),(0,10),(1,10),(1,10),(1,10),(0,1), (0,10),(0,10),(0,10),(11,40),(11,40),(11,40),(0,1)])
    print(HC_params[0])
    cash = bot_testing('complex', HC_params[0])
    results['HC'].append(cash)

print(results)
