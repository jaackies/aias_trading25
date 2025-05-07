import numpy as np


# def pad(P, N):
#     # where P is your array of data points
#     # and N is the # data pt.
#     padding = -np.flip(P[1:N])
#     return np.append(padding, P)


def sma_filter(N: int) -> np.ndarray:
    return np.ones(N) / N


def lma_filter(N: int) -> np.ndarray:
    # return (np.full(N, 2) / (N + 1)) * (1 - (np.arange(N) / N))
    # return (np.full(N, 2) - (np.arange(N) / N)) / (N + np.ones(N))
    w = np.arange(1, N + 1)
    return w / w.sum()


def ema_filter(N: int, sf: float) -> np.ndarray:
    # TODO: see https://arc.net/l/quote/cjtoaouc
    assert sf > 0, f"Smoothing factor must be between 0 and inf. (sf = {sf})"
    # w = sf * ((1 - sf) ** np.arange(N))
    # return w / np.sum(w)  # Normalize the weights
    # return np.full(N, sf) * np.power((np.ones(N) - np.full(N, sf)), np.arange(N))
    w = sf * ((1 - sf) ** np.arange(N))
    return w / w.sum()


def wma_signal(P: np.ndarray, kernel: np.ndarray):
    N = len(kernel)
    padded_P = np.concat((np.flip(P[1:N]), P))
    kernel /= kernel.sum()
    return np.convolve(padded_P, np.flip(kernel), "valid")

    # P = array of data points
    # N = number of days
    # kernel = filter method called on N
    # return np.convolve(pad(P, N), kernel, "valid")


def holding_signal(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """True if high freq signal is above low freq signal (i.e. whether to hold or not)."""
    return high > low


def complex_signal(data, w1, w2, w3, d1, d2, d3, sf):
    """sma, lma, ema"""
    sma = wma_signal(data, sma_filter(d1)) * w1 if w1 else 0
    lma = wma_signal(data, lma_filter(d2)) * w2 if w2 else 0
    ema = wma_signal(data, ema_filter(d3, sf)) * w3 if w3 else 0
    return (sma + lma + ema) / (w1 + w2 + w3)
