import bot
import pandas as pd
import problem as fitness

close_price = pd.read_csv('training.csv')['close']
print(close_price)
x = fitness.bot_fitness_func(9, 35)
print(x)