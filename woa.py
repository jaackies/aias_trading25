import numpy as np
import pandas as pd

def whale_optimization(fitness_func, bot_type, dim, bounds, num_agents, max_iter, integer_dims):
    a = 2
    history = []
    Init_whalses = np.array([np.random.uniform(low, high, num_agents) for (low, high) in bounds]).T #Initialize the whales positions (which are intial solutions)
    Int_whales = np.array([convert_array_for_fitness(individual, integer_dims) for individual in Init_whalses])
    print(integer_dims)
    best = Int_whales[0].copy() #initialize optimal solution
    print(best)
    params = convert_array_for_fitness(best, integer_dims)
    best_score = evaluate_function(fitness_func, bot_type, params)
    for i in range(num_agents): # compare to find the current optimal solution
        params = convert_array_for_fitness(Int_whales[i], integer_dims)
        score = evaluate_function(fitness_func, bot_type, params)
        if score > best_score:
            best = Int_whales[i].copy()
            best_score = score

    for t in range(max_iter): # while (t < maximum number of iterations)
        a = 2 - t * (2 / max_iter) # calculate a. make sure a is linearly decreased from 2 to 0

        for i in range(num_agents):
            r = np.random.rand(dim)
            A = 2 * a * r - a # Update A
            C = 2 * r # Update C
            p = np.random.rand()

            if p < 0.5: # if p < 0.5
                if np.linalg.norm(A) >= 1: # Exploration phase
                    rand_index = np.random.randint(0, num_agents)
                    X_rand = Int_whales[rand_index]
                    D = np.abs(C * X_rand - Int_whales[i])
                    Int_whales[i] = X_rand - A * D
                else: # Exploitation phase - Shrinking encircling mechanism
                    D = np.abs(C * best - Int_whales[i])
                    Int_whales[i] = best - A * D 
            else: #if p >= 0.5；Exploitation phase - Spiral updating position
                D = np.abs(best - Int_whales[i]) 
                b = 1
                l = np.random.uniform(-1, 1, dim) # Update l
                Int_whales[i] = D * np.exp(b * l) * np.cos(2 * np.pi * l) + best

            # Check if any search agent goes beyond the search space and amend it
            for d in range(dim):
                Int_whales[i][d] = np.clip(Int_whales[i][d], bounds[d][0], bounds[d][1])
                if integer_dims and d in integer_dims:
                    Int_whales[i][d] = int(round(Int_whales[i][d]))
            params = convert_array_for_fitness(Int_whales[i], integer_dims)
            score = evaluate_function(fitness_func, bot_type, params)
            row = [t,i] + list(Int_whales[i]) + [score, best_score]
            history.append(row)
            if score > best_score: # Update X* is there is a better solution
                best = Int_whales[i].copy()
                best_score = score
    # Generate csv file
    columns = ["Iteration", "Whale_ID"] + [f"Dim_{j}" for j in range(dim)] + ["Fitness", "Best_So_far"]
    df = pd.DataFrame(history, columns=columns)
    df.to_csv("woa_whale_log.csv", index=False)
    return best, best_score


#  Round and convert the specified columns of the array to integers
#  integer_dims: the colums need to be converted
#     return result
def convert_array_for_fitness(X_row, integer_dims):
    result = []
    for i, val in enumerate(X_row):
        if i in integer_dims:
            result.append(int(round(val)))  # 强制转 int
        else:
            result.append(float(val))       # 保留 float
    return result

def evaluate_function(fitness_func, bot_type, params):
    if bot_type == "complex":
        high = params [:7]
        low = params [7:]
        score = fitness_func(bot_type, high, low)
    else:
        score = fitness_func(bot_type, *params)
    return score