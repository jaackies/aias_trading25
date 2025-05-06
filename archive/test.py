import bot
import pandas as pd
import test_problem as fitness
import visual

close_price = pd.read_csv("training.csv")["close"]
print(close_price)
x = fitness.bot_fitness_func("sma", 10, 20)
print(x)
# visual.plot_sma_cross(close_price, short_window=10, long_window=30)
