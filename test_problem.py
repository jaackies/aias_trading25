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
training = daily_df[daily_df["date"]<"2020-01-01"].iloc[::-1]
testing = daily_df[daily_df["date"]>="2020-01-01"].iloc[::-1]
# convert the data into csv files
training.to_csv('training.csv')
testing.to_csv('testing.csv')

def bot_testing(bot_type, high_window, low_window, alpha=0):
    # This function will calculate the fitness (total cash earned from the buy/sell signals) of the trading bot, including the time of the transactions made
    # initial values
    cash = 1000
    fee=0.03
    bitcoin = 0.0

    # get the data
    close_price = pd.read_csv('testing.csv')['close']
    time = pd.read_csv('testing.csv')['date']

    # the list to save the transaction history
    result=[]
    bot_signals=[]

    # intialise bot, use training dataset for optimisation algorithms fitness functions
    if bot_type.lower() == 'sma' and alpha ==0:
        bot_signals = bot.get_signals_sma2(close_price, high_window, low_window)
    elif bot_type.lower() == 'smaema' and alpha!=0:
        bot_signals = bot.get_signals_smaema(close_price, low_window, high_window, alpha)
    elif bot_type.lower() == 'complex' and alpha==0: #as alpha values were stored in the list of high_window and low_window
        bot_signals = bot.get_signals_complex(close_price, high_window, low_window)
    
    #loop through the time length
    for i in range(min(len(bot_signals), len(close_price)-1)):
        close=close_price[i]
        # buy
        if bot_signals[i] == "buy":
            bitcoin =  (cash*(1-fee))/close
            cash = 0
            result.append([time[i],cash, bitcoin])
        # sell
        elif bot_signals[i] == "sell":
            cash = bitcoin * close * (1-fee)
            bitcoin =0
            result.append([time[i],cash, bitcoin])
    
    # final evaluation to change back to cash
    last_close=close_price.iloc[-1]
    if bitcoin>0:
        cash = bitcoin * last_close * (1-fee)
        bitcoin =0
        result.append([time[i],cash, bitcoin])
    
    return result

def bot_evaluation(bot_type, high_window, low_window, alpha=0):
    # This function will returns the result nicely
    result_lst=bot_testing(bot_type, high_window, low_window, alpha)
    result_df=pd.DataFrame(result_lst, columns=["Time", "Cash", "Bitcoin"])
    print(result_df.to_string(index=False, justify="center", float_format='{:,.2f}'.format))
    return result_df