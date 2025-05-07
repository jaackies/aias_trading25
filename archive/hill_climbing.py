import numpy as np

def hill_climbing(bot_type, high_frequency_window, low_frequency_window, alpha=0, max_iter=100):
    high_window = high_frequency_window
    low_window = low_frequency_window
    new_high_frequency_window = high_frequency_window
    new_low_frequency_window= low_frequency_window
    alpha=alpha
    new_alpha=alpha
    cash1=0
    for i in range(max_iter):
        # find the correct parameter tweak:
        if bot_type.lower() == 'sma':
            new_high_frequency_window, new_low_frequency_window = parameter_tweak('sma', high_window, low_window)
        elif bot_type.lower() == 'smaema' and alpha !=0:
            new_high_frequency_window, new_low_frequency_window, new_alpha = parameter_tweak('smaema', high_window, low_window, alpha)
        elif bot_type.lower() == 'complex':
            new_high_frequency_window, new_low_frequency_window = parameter_tweak('complex',high_window, low_window)
            
        # run the total cash return after trading
        cash1=bot_fitness_func(bot_type, high_window, low_window, alpha)
        cash2=bot_fitness_func(bot_type, new_high_frequency_window, new_low_frequency_window, new_alpha)
        # compare the cash earned after tweaking the parameters
        if cash2 > cash1:
            high_window = new_high_frequency_window
            low_window = new_low_frequency_window
            alpha=new_alpha
            cash1 = cash2
    if bot_type.lower() == 'sma' or bot_type.lower() == 'complex':
        return high_window, low_window, float(cash1)
    elif bot_type.lower() == 'smaema':
        return high_window, low_window, alpha, float(cash1)

def parameter_tweak(bot_type, hfw, lfw, alpha =0):
    if bot_type.lower() == 'sma':
        new_hfw, new_lfw = window_tweak(hfw, lfw)
        return new_hfw, new_lfw
    elif bot_type.lower() == 'smaema' and alpha !=0:
        new_hfw, new_lfw = window_tweak(hfw, lfw)
        new_alpha = alpha_tweak(alpha)
        return new_hfw, new_lfw, new_alpha
    elif bot_type.lower() == 'complex':
        new_hfw, new_lfw = complex_tweak(hfw, lfw)
        return new_hfw, new_lfw

def window_tweak(hfw, lfw):
    rng = np.random.default_rng()
    for _ in range(100):
        # new high frequency window
        a=int(rng.integers(-5,6))
        new_hfw=hfw+a
        # new low frequency window
        b=int(rng.integers(-5,6))
        new_lfw=lfw+b
        # we check to make sure that new_hfw in range(11,40) and new_lfw(2,10)
        new_hfw = max(1, min(10, new_hfw))
        new_lfw = max(11, min(40, new_lfw))
        if new_hfw < new_lfw:
            return new_hfw, new_lfw
    return hfw, lfw

def alpha_tweak(alpha):
    rng = np.random.default_rng()
    for _ in range(100):
        diff = rng.uniform(-0.15, 0.15)
        new_alpha = alpha + diff
        if 0.1<= new_alpha <= 1:
            return new_alpha
    return alpha # if after 100 loops and cannot find the optimal value, then we return the original alpha

def weights_tweak(weight_lst):
    rng = np.random.default_rng()
    # random index number of a weight in a list
    position=int(rng.integers(0,3))
    # normalised weights
    normalised_weights = [float(weight)/sum(weight_lst) for weight in weight_lst]

    # tweak the randomly selected weight with proportion different
    selected_weight_rate = normalised_weights[position]
    diff = np.random.uniform(-0.15, 0.15)
    selected_weight_rate += diff

    # the rate of other weights
    other_weight_idx= [idx for idx in range(3) if idx!=position]
    remainder_rate = 1- selected_weight_rate 
    # the remaining 2 weights, we generate randomly the proportion and multiply it with the current rates 
    one_weight_rate = rng.uniform(0.1, 0.9)
    another_weight_rate = 1- one_weight_rate
    other_weight_rates=[one_weight_rate, another_weight_rate]
    new_other_weight_rates = [remainder_rate * s for s in other_weight_rates]

    new_weights = [0,0,0]
    new_weights[position] = selected_weight_rate * sum(weight_lst)
    for i in range(2):
        idx = other_weight_idx[i]
        new_weights[idx] = new_other_weight_rates[i] * sum(weight_lst)

    return new_weights[0], new_weights[1], new_weights[2]


def complex_tweak(hfw, lfw):
    # in this code, hfw and lfw input are array-type in format of [w1, w2, w3, d1, d2, d3, sf]
    # weight tweak
    new_hfw_w1, new_hfw_w2,new_hfw_w3 = weights_tweak(hfw[0:3])
    new_lfw_w1, new_lfw_w2,new_lfw_w3 = weights_tweak(lfw[0:3])

    # window tweak
    new_hfw_d1, new_lfw_d1 = window_tweak(hfw[3], lfw[3])
    new_hfw_d2, new_lfw_d2 = window_tweak(hfw[4], lfw[4])
    new_hfw_d3, new_lfw_d3 = window_tweak(hfw[5], lfw[5])

    # alpha tweak
    new_hfw_alpha = alpha_tweak(hfw[-1])
    new_lfw_alpha = alpha_tweak(lfw[-1])

    return [new_hfw_w1, new_hfw_w2, new_hfw_w3, new_hfw_d1, new_hfw_d2, new_hfw_d3,new_hfw_alpha], [new_lfw_w1, new_lfw_w2, new_lfw_w3, new_lfw_d1, new_lfw_d2, new_lfw_d3, new_lfw_alpha]

