import pandas as pd
import bot

def bot_testing(bot_type, optimal_values):
    # intialise bot, use training dataset for optimisation algorithms fitness functions
    bot_signals=[]
    close_price = pd.read_csv('testing.csv')['close']

    # intialise bot, use training dataset for optimisation algorithms fitness functions
    if bot_type.lower() == 'sma':
        bot_signals = bot.get_signals_sma2(close_price, optimal_values[0], optimal_values[1])
    elif bot_type.lower() == 'smaema':
        bot_signals = bot.get_signals_smaema(close_price, optimal_values[0], optimal_values[1], optimal_values[2])
    elif bot_type.lower() == 'complex': #as alpha values were stored in the list of high_window and low_window
        bot_signals = bot.get_signals_complex(close_price, optimal_values[0], optimal_values[1])
    
    # initial values
    cash = 1000
    fee=0.03
    bitcoin = 0.0
    
    #loop through the time length
    for i in range(min(len(bot_signals),len(close_price)-1)):
        close=close_price[i]
        # buy
        if bot_signals[i] == "buy" and cash>0:
            bitcoin =  (cash*(1-fee))/close
            cash = 0
        # sell
        elif bot_signals[i] == "sell" and bitcoin>0:
            cash = bitcoin * close * (1-fee)
            bitcoin =0
    
    # final evaluation to change back to cash
    last_close=close_price.iloc[-1]
    if bitcoin>0:
        cash = bitcoin * last_close * (1-fee)
        bitcoin =0
        return cash
    elif cash>0:
        return cash