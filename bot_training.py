import pandas as pd
import bot

def bot_training(bot_type, high_window, low_window, alpha): # change bot_signals to high_frequency_window_size, low_frequency_window_size
    # intialise bot, use training dataset for optimisation algorithms fitness functions
    bot_signals=[]
    close_price = pd.read_csv('training.csv')['close']
    
    if bot_type.lower() == 'sma':
        bot_signals = bot.get_signals_sma2(close_price, high_window, low_window) #need to think about how to call 2 other bot algorithms
    elif bot_type.lower() == 'smaema':
        bot_signals = bot.get_signals_smaema(close_price, high_window, low_window, alpha)
    elif bot_type.lower() == 'complex': #as alpha values were stored in the list of high_window and low_window
        bot_signals = bot.get_signals_complex(close_price, high_window, low_window)

    # initial values
    cash = 1000
    fee=0.03
    bitcoin = 0.0

    #loop through the time length
    for i in range(min(len(bot_signals),len(close_price)-1)):
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
    elif cash >0:
        return cash