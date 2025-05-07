import numpy as np
from bot_training import bot_training

# Group A: 
# High(1-10), Low(11-40)
# sma: [(1,10),(11,40)]
# smaema: [(1,10),(11,40),(0,1)]
# weights: [(0,10),(0,10),(0,10),(1,10),(1,10),(1,10),(0,1), (0,10),(0,10),(0,10),(11,40),(11,40),(11,40),(0,1)]

# Group B:
# High(5-50), Low(51-100)
# sma: [(5,50),(51,100)]
# smaema: [(5,50),(51,100),(0,1)]
# weights: [(0,10),(0,10),(0,10),(5,50),(5,50),(5,50),(0,1), (0,10),(0,10),(0,10),(51,100),(51,100),(51,100),(0,1)]

def hill_climbing(bot_type, bounds, max_iter=1000):
    rng = np.random.default_rng()
    high_window = None
    low_window = None
    alpha=0
    new_high_frequency_window = high_window 
    new_low_frequency_window = low_window
    new_alpha=0
    # processing the bounds inputs
    if bot_type.lower() == 'sma':
        high_window = int(rng.integers(bounds[0][0],bounds[0][1]))
        low_window = int(rng.integers(bounds[1][0],bounds[1][1]))
    elif bot_type.lower() == 'smaema':
        high_window = int(rng.integers(bounds[0][0],bounds[0][1]))
        low_window = int(rng.integers(bounds[1][0],bounds[1][1]))
        alpha = rng.uniform(bounds[2][0], bounds[-1][1])
    elif bot_type.lower() == 'complex':
        # generate the values for high
        weight_sma_high= int(rng.integers(bounds[0][0],bounds[0][1]))
        weight_lma_high= int(rng.integers(bounds[1][0],bounds[0][1]))
        weight_ema_high= int(rng.integers(bounds[2][0],bounds[0][1]))
        window_sma_high= int(rng.integers(bounds[3][0],bounds[0][1]))
        window_lma_high= int(rng.integers(bounds[4][0],bounds[0][1]))
        window_ema_high= int(rng.integers(bounds[5][0],bounds[0][1]))
        alpha_high = rng.uniform(bounds[6][0], bounds[6][1])
        high_window = [weight_sma_high,weight_lma_high, weight_ema_high, window_sma_high, window_lma_high, window_ema_high, alpha_high]
        # generate the values for low
        weight_sma_low= int(rng.integers(bounds[7][0],bounds[7][1]))
        weight_lma_low= int(rng.integers(bounds[8][0],bounds[8][1]))
        weight_ema_low= int(rng.integers(bounds[9][0],bounds[9][1]))
        window_sma_low= int(rng.integers(bounds[10][0],bounds[10][1]))
        window_lma_low= int(rng.integers(bounds[11][0],bounds[11][1]))
        window_ema_low= int(rng.integers(bounds[12][0],bounds[12][1]))
        alpha_low = rng.uniform(bounds[13][0], bounds[13][1])
        low_window = [weight_sma_low, weight_lma_low, weight_ema_low, window_sma_low, window_lma_low, window_ema_low, alpha_low]

    # the best cash
    cash1=0
    for i in range(max_iter):
        # find the correct parameter tweak:
        if bot_type.lower() == 'sma':
            new_high_frequency_window, new_low_frequency_window = window_tweak(high_window, low_window, bounds[0], bounds[1])
        elif bot_type.lower() == 'smaema':
            new_high_frequency_window, new_low_frequency_window = window_tweak(high_window, low_window, bounds[0], bounds[1])
            new_alpha = alpha_tweak(alpha, bounds[2])
        elif bot_type.lower() == 'complex':
            # generate the values 
            new_high_frequency_window, new_low_frequency_window = complex_tweak(high_window, low_window, bounds)
            
        # run the total cash return after trading
        cash1=bot_training(bot_type, high_window, low_window, alpha)
        cash2=bot_training(bot_type, new_high_frequency_window, new_low_frequency_window,alpha)
        # compare the cash earned after tweaking the parameters
        if cash2 > cash1:
            high_window = new_high_frequency_window
            low_window = new_low_frequency_window
            alpha=new_alpha
            cash1 = cash2
    if bot_type.lower() == 'sma' or bot_type.lower() == 'complex':
        return [high_window, low_window], float(cash1)
    elif bot_type.lower() == 'smaema':
        return [high_window, low_window, alpha], float(cash1)

def window_tweak(hfw, lfw, window_range1, window_range2):
    rng = np.random.default_rng()
    for _ in range(100):
        # new high frequency window
        a=int(rng.integers(-5,6))
        new_hfw=hfw+a
        # new low frequency window
        b=int(rng.integers(-5,6))
        new_lfw=lfw+b
        # we check to make sure that new_hfw in range(11,40) and new_lfw(2,10)
        new_hfw = max(window_range1[0], min(window_range1[1], new_hfw)) # can change the bounds to 5-50 
        new_lfw = max(window_range2[0], min(window_range2[1], new_lfw)) # can change the bounds to 51-100
        if new_hfw < new_lfw:
            return new_hfw, new_lfw
    return hfw, lfw

def alpha_tweak(alpha, alpha_range):
    rng = np.random.default_rng()
    for _ in range(100):
        diff = rng.uniform(-0.15, 0.15)
        new_alpha = alpha + diff
        if alpha_range[0]<= new_alpha <= alpha_range[1]:
            return new_alpha
    return alpha # if after 100 loops and cannot find the optimal value, then we return the original alpha

def weights_tweak(weight_lst, weight_range):
    rng = np.random.default_rng()
    weights=[]
    for i in range(len(weight_range)):
        weight = weight_lst[i]
        diff = rng.uniform(-1.5, 1.5)
        weight += diff
        new_weight = max(weight_range[i][0], min(weight_range[i][1], weight))
        weights.append(new_weight)
    return weights


def complex_tweak(hfw, lfw, bounds):
    # in this code, hfw and lfw input are array-type in format of [w1_h, w2_h, w3_h, d1_h, d2_h, d3_h, sf_h, w1_l, w2_l, w3_l, d1_l, d2_l, d3_l, sf_l]
    # weight tweak

    new_hfw_w1, new_hfw_w2,new_hfw_w3 = weights_tweak(hfw[0:3], bounds[0:3])
    new_lfw_w1, new_lfw_w2,new_lfw_w3 = weights_tweak(lfw[0:3], bounds[7:10])

    # window tweak
    new_hfw_d1, new_lfw_d1 = window_tweak(hfw[3], lfw[3], bounds[3], bounds[10])
    new_hfw_d2, new_lfw_d2 = window_tweak(hfw[4], lfw[4], bounds[4], bounds[11])
    new_hfw_d3, new_lfw_d3 = window_tweak(hfw[5], lfw[5], bounds[5], bounds[12])

    # alpha tweak
    new_hfw_alpha = alpha_tweak(hfw[-1], bounds[6])
    new_lfw_alpha = alpha_tweak(lfw[-1], bounds[13])

    return [new_hfw_w1, new_hfw_w2, new_hfw_w3, new_hfw_d1, new_hfw_d2, new_hfw_d3,new_hfw_alpha],[new_lfw_w1, new_lfw_w2, new_lfw_w3, new_lfw_d1, new_lfw_d2, new_lfw_d3, new_lfw_alpha]
