import numpy as np

# UTILS


def sma_filter(N: int) -> np.ndarray:
    return np.ones(N)


def lma_filter(N: int) -> np.ndarray:
    return (np.full(N, 2) - (np.arange(N) / N)) / (N + np.ones(N))


def ema_filter(N: int, sf: float) -> np.ndarray:
    # sf = alpha = smoothing factor
    return np.full(N, sf) * np.power((np.ones(N) - np.full(N, sf)), np.arange(N))


def pad(P: np.ndarray, N: int) -> np.ndarray:
    # where P is your numpy array of data points
    # and N is the window size
    padding = -np.flip(P[1:N])
    return np.concatenate((padding, P))


# SIGNALS


def wma_signal(P: np.ndarray, kernel: np.ndarray):
    N = len(kernel)
    kernel /= kernel.sum()
    return np.convolve(pad(P, N), np.flip(kernel), "valid")


def buysell_signals(high_signal: np.ndarray, low_signal: np.ndarray) -> np.ndarray:
    # obtains buy/sell signals from a high frequency signal and a low frequency signal
    # returns python array continaing strings of "buy", "sell" or "none"
    difference = high_signal - low_signal
    signs = np.sign(difference)
    shifted = np.roll(signs, 1)
    shifts = signs != shifted
    pos_filter = np.logical_and(shifts, signs == 1)
    neg_filter = np.logical_and(shifts, signs == -1)
    final_signals = np.full(signs.shape, "none")
    final_signals[pos_filter] = "buy"
    final_signals[neg_filter] = "sell"
    return final_signals


def complex_signal(data: np.ndarray, w1, w2, w3, d1, d2, d3, sf):
    """sma, lma, ema"""
    summed_weights = w1 + w2 + w3
    if summed_weights == 0:
        return np.zeros(data.shape)
    sma = wma_signal(data, sma_filter(d1)) * w1 if w1 > 0 else 0
    lma = wma_signal(data, lma_filter(d2)) * w2 if w2 > 0 else 0
    ema = wma_signal(data, ema_filter(d3, sf)) * w3 if w3 > 0 else 0
    out = (sma + lma + ema) / summed_weights
    if out is np.nan:
        print(sma, lma, ema)
    return out


# BOT SIGNALS


def sma2_bot_signal(data, highN, lowN):
    # get buy and sell signals by using two SMA filters
    # one with window size highN that is small for high freq signal
    # one with window size lowN that is larger for low freq signal
    high_signal = wma_signal(data, sma_filter(highN))
    low_signal = wma_signal(data, sma_filter(lowN))
    return buysell_signals(high_signal, low_signal)


def smaema_bot_signal(data, EN, lowN, Esf):
    # get buy and sell signals by using a
    # SMA filter for low freq (lowN = larger window size) and
    # EMA for high frequency (EN = smaller window size, Esf = smoothing factor)
    high_signal = wma_signal(data, ema_filter(EN, Esf))
    low_signal = wma_signal(data, sma_filter(lowN))
    return buysell_signals(high_signal, low_signal)


def complex_bot_signal(data, *params):
    high, low = params[:7], params[7:]
    # high and low are arrays [w1, w2, w3, d1, d2, d3, sf], defining parameters of
    # weight, length of windows and smoothing factors of the complex frequency
    # high is the array defining the paramters for high frequency
    # low is the array defining the parameteres for low frequency
    high_signal = complex_signal(data, *high)
    low_signal = complex_signal(data, *low)
    return buysell_signals(high_signal, low_signal)
