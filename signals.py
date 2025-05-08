import numpy as np


# def pad(P, N):
#     # where P is your array of data points
#     # and N is the # data pt.
#     padding = -np.flip(P[1:N])
#     return np.append(padding, P)


def sma_filter(N: int) -> np.ndarray:
    return np.ones(N) / N


def lma_filter(N: int) -> np.ndarray:
    # w = np.arange(1, N + 1)
    # return w / w.sum()
    return (np.full(N, 2) - (np.arange(N) / N)) / (N + np.ones(N))


def ema_filter(N: int, sf: float) -> np.ndarray:
    # TODO: see https://arc.net/l/quote/cjtoaouc
    # assert sf > 0, f"Smoothing factor must be between 0 and inf. (sf = {sf})"
    # w = sf * ((1 - sf) ** np.arange(N))
    # return w / w.sum()
    return np.full(N, sf) * np.power((np.ones(N) - np.full(N, sf)), np.arange(N))


def pad(P, N):
    # where P is your numpy array of data points
    # and N is the window size
    padding = -np.flip(P[1:N])
    return np.append(padding, P)


def wma_signal(P, kernel):
    # P = array of data points
    # N = window size
    # kernel = filter method called on N
    N = len(kernel)
    return np.convolve(pad(P, N), kernel, "valid")


# def wma_signal(P: np.ndarray, kernel: np.ndarray):
#     N = len(kernel)
#     padded_P = np.concat((np.flip(P[1:N]), P))
#     kernel /= kernel.sum()
#     return np.convolve(padded_P, np.flip(kernel), "valid")


# def holding_signal(high: np.ndarray, low: np.ndarray) -> np.ndarray:
#     """True if high freq signal is above low freq signal (i.e. whether to hold or not)."""
#     return high > low


def buysell_signals(high_signal, low_signal):
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
    return final_signals.tolist()


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
