import numpy as np
import pandas as pd


# def sim_bot(holding_signal: np.ndarray, price: np.ndarray, fee=0.03, cash=1000):
#     # Ensures that the bot cashes out at the end of the simulation
#     holding_signal[-1] = False

#     indicies = np.diff(holding_signal, prepend=False).nonzero()[0]
#     fee_factor = 1 - fee
#     units = cash

#     for i in indicies:
#         units *= fee_factor
#         if holding_signal[i]:  #  buy
#             units /= price[i]
#         else:  # sell
#             units *= price[i]

#     return units


def sim_bot(buysell_signals, price, fee=0.03, cash=1000):
    bot_signals = buysell_signals
    close_price = pd.Series(price)

    # initial values
    bitcoin = 0.0

    # loop through the time length
    for i in range(min(len(bot_signals), len(close_price) - 1)):
        close = close_price[i]
        # buy
        if bot_signals[i] == "buy" and cash > 0:
            bitcoin = (cash * (1 - fee)) / close
            cash = 0
        # sell
        elif bot_signals[i] == "sell" and bitcoin > 0:
            cash = bitcoin * close * (1 - fee)
            bitcoin = 0

    # final evaluation to change back to cash
    last_close = close_price.iloc[-1]
    if bitcoin > 0:
        cash = bitcoin * last_close * (1 - fee)
        bitcoin = 0
        return cash
    elif cash > 0:
        return cash
