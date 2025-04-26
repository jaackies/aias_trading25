import numpy as np

def whale_optimization(fitness_func, dim, bounds, num_agents, max_iter):
    a = 2
    X = np.random.uniform(bounds[0], bounds[1], (num_agents, dim)) #Initialize the whales position
    # np.random.uniform(a,b,size) Sample random values from the interval [a,b), and generate an array with the shape specified by size(like: num_agents rows; dim columns)
    best = X[0].copy()
    best_score = fitness_func(best)
    for i in range(num_agents): # initialize the whales position(soluntions), compare to find the current optimal solution
        score = fitness_func(X[i])
        if score < best_score:
            best = X[i].copy()
            best_score = score

    for t in range(max_iter): # while (t < maximum number of iterations)
        a = 2 - t * (2 / max_iter) # calculate a. make sure a is linearly decreased from 2 to 0
        print("Iteration：" + str(t))

        for i in range(num_agents):
            r = np.random.rand(dim)
            A = 2 * a * r - a # Update A
            C = 2 * r # Update C

            if np.random.rand() < 0.5: # if p < 0.5
                if np.linalg.norm(A) >= 1: # Exploration phase
                    rand_index = np.random.randint(0, num_agents)
                    X_rand = X[rand_index]
                    D = np.abs(C * X_rand - X[i])
                    X[i] = X_rand - A * D
                else: # Exploitation phase - Shrinking encircling mechanism
                    D = np.abs(C * best - X[i])
                    X[i] = best - A * D 
            else: #if p >= 0.5；Exploitation phase - Spiral updating position
                D = np.abs(best - X[i]) 
                b = 1
                l = np.random.uniform(-1, 1, dim) # Update l
                X[i] = D * np.exp(b * l) * np.cos(2 * np.pi * l) + best

            X[i] = np.clip(X[i], bounds[0], bounds[1]) # Check if any search agent goes beyond the search space and amend it
            score = fitness_func(X[i]) #Calculate the fitness of each search agent
            print("Whales " + str(i) + ": ")
            print("Solution: " + str(X[i]) + ", " + "score: " + str(score))
            if score < best_score: # Update X* is there is a better solution
                best = X[i].copy()
                best_score = score

    return best, best_score