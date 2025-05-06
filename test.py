import pandas as pd
import problem as fitness

close_price = pd.read_csv('training.csv')['close']
print(close_price)
X = fitness.bot_fitness_func("smaema", 5, 30, 0.2)
print(X)

