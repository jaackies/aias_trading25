import numpy as np
import pandas as pd


def sim_bot(bot_signals, price, fee=0.03, cash=1000):
    indicies = np.nonzero(bot_signals != "none")[0]

    fee_factor = 1 - fee
    units = cash

    if len(indicies) == 0:
        return cash

    if bot_signals[indicies[0]] == "sell":  # then buy at the start
        units *= fee_factor
        units /= price[indicies[0]]

    for i in indicies:
        units *= fee_factor
        if bot_signals[i] == "buy":
            units /= price[i]
        elif bot_signals[i] == "sell":
            units *= price[i]

    if bot_signals[indicies[-1]] == "buy":  # then sell at the end
        units *= fee_factor
        units *= price[-1]

    return units
