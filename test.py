import pandas as pd
import test_problem as fitness

close_price = pd.read_csv('testing.csv')['close']
print(close_price)
X = fitness.bot_evaluation("complex", [5.4953370921131155, 8.382614021366823, 7.153990751589008, 49, 43, 20, 1.0], [7.125885056101225, 7.402070155679087, 7.123167725420541, 90, 37, 2, 0.6681183259029129])
print(X)

