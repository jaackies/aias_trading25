# import numpy as np 
# def sphere(x):
#     return np.sum(x ** 2)

# import important library
import pandas as pd
import numpy as np
import bot
import importlib
importlib.reload(bot)

# read the csv file
daily_df=pd.read_csv('BTC-Daily.csv')
daily_df.head()
# convert time strings into timestamps, and take only year, month and date values
daily_df["date"]=pd.to_datetime(daily_df["date"]).dt.strftime("%Y-%m-%d")
daily_df.shape
# splitting the data
training = daily_df[daily_df["date"]<"2020-01-01"]
testing = daily_df[daily_df["date"]>="2020-01-01"]
# convert the data into csv files
training.to_csv('training.csv')
testing.to_csv('testing.csv')
print(training.shape)
print(testing.shape)


def bot_fitness_func(bot_type, high_window, low_window, alpha=0): # change bot_signals to high_frequency_window_size, low_frequency_window_size
    """
    This function will calculate the fitness (total cash earned from the buy/sell signals) of the trading bot
    Parameters:
    - 
    - 
    Return:
    - 
    """ 
    
    # call the bot in here
    # Jackie looks after this part - just need to call the bot with 2 parameters (high_frequency_window_size, low_frequency_window_size)
    
    # intialise bot, use training dataset for optimisation algorithms fitness functions
    bot_signals=[]
    close_price = pd.read_csv('training.csv')['close']
    
    if bot_type.lower() == 'sma' and alpha ==0:
        bot_signals = bot.get_signals_sma2(close_price, high_window, low_window) #need to think about how to call 2 other bot algorithms
    elif bot_type.lower() == 'smaema' and alpha!=0:
        bot_signals = bot.get_signals_smaema(close_price, high_window, low_window, alpha)
    elif bot_type.lower() == 'complex' and alpha==0: #as alpha values were stored in the list of high_window and low_window
        bot_signals = bot.get_signals_complex(close_price, high_window, low_window)

    # initial values
    cash = 1000
    fee=0.03
    bitcoin = 0.0

    #loop through the time length
    for i in range(len(close_price)-1):
        close=close_price.iloc[i]
        # buy, ensure we have cash to buy
        if bot_signals[i] == "buy" and cash>0:
            bitcoin =  (cash*(1-fee))/close
            cash = 0
        # sell, ensure we have bitcoin to sell
        elif bot_signals[i] == "sell" and bitcoin>0:
            cash = bitcoin * close * (1-fee)
            bitcoin =0
    
    # final evaluation to change back to cash
    last_close=close_price.iloc[-1]
    if bitcoin>0:
        cash = bitcoin * last_close * (1-fee)
        bitcoin =0
    
    return cash