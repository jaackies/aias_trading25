import numpy as np


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
