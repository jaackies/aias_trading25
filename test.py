import pandas as pd
import test_problem as fitness

close_price = pd.read_csv('training.csv')['close']
print(close_price)
X = fitness.bot_fitness_func("sma", 5, 50)
print(X)

