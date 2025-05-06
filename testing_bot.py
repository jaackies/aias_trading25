from bot import *
import numpy as np

# this python file serves as example usage of the bot, using dummy data

raw_data = [0, 0.25, 0.5, 0.8, 0.8, 0.55, 0.52, 0.51, 0.6, 0.61,
            0.65, 0.8, 0.9, 1, 1.25, 1.4, 1.5, 1.7, 1.7, 1.8, 1.81,
            1.82, 1.82, 1.9, 2, 2, 1.99, 1.95, 1.7, 1.5, 1,3, 1.2, 1.2,
            1.3, 1.4, 1.45, 1.4, 1.25, 1, 0.75, 0.55, 0.5, 0.5, 0.5, 0.5,
            0.45, 0.4, 0.3, 0.25, 0.2, 0.1, 0, -0.25, -0.3, -0.4, -0.5]
data = np.array(raw_data)

# setting param arrays for use in complex signals
high_params = [1, 2, 2, 3, 4, 5, 0.5]
low_params = [2, 2, 2, 10, 8, 9, 0.2]

# note that for ease of viewing, i've sliced only the
# first 15 elements of the resulting buy/sell array
print(get_signals_sma2(data, 2, 6)[:15])
print(get_signals_smaema(data, 2, 6, 0.4)[:15])
print(get_signals_complex(data, high_params, low_params)[:15])