import numpy as np


def sim_bot(holding_signal: np.ndarray, price: np.ndarray, fee=0.03, cash=1000):
    holding_signal[-1] = (
        False  # Ensures that the bot cashes out at the end of the simulation
    )

    indicies = np.diff(holding_signal, prepend=False).nonzero()[0]
    fee_factor = 1 - fee
    units = cash

    for i in indicies:
        units *= fee_factor
        if holding_signal[i]:  #  buy
            units /= price[i]
        else:  # sell
            units *= price[i]

    return units
