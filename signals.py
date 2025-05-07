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


def ema_filter(sf, N):
    # w = sf * ((1 - sf) ** np.arange(N))
    # return w / np.sum(w)  # Normalize the weights
    # return np.full(N, sf) * np.power((np.ones(N) - np.full(N, sf)), np.arange(N))
    w = sf * ((1 - sf) ** np.arange(N))
    return w / w.sum()


def wma(P: np.ndarray, kernel: np.ndarray):
    N = len(kernel)
    padded_P = np.concat((np.flip(P[1:N]), P))
    kernel /= kernel.sum()
    return np.convolve(padded_P, np.flip(kernel), "valid")

    # P = array of data points
    # N = number of days
    # kernel = filter method called on N
    # return np.convolve(pad(P, N), kernel, "valid")


def diff(short_signal: np.ndarray, long_signal: np.ndarray) -> np.ndarray:
    return short_signal - long_signal


def buy_sell(signal: np.ndarray) -> np.ndarray:
    """Buy when 1 and sell when -1."""
    output = np.zeros_like(signal)
    output[1:] = np.where((signal[:-1] > 0) & (signal[1:] <= 0), 1, output[1:])
    output[1:] = np.where((signal[:-1] <= 0) & (signal[1:] > 0), -1, output[1:])
    return output


def complex_signal(data, w1, w2, w3, d1, d2, d3, sf):
    sma = wma(data, sma_filter(d1)) * w1
    lma = wma(data, lma_filter(d2)) * w2
    ema = wma(data, ema_filter(sf, d3)) * w3
    return (sma + lma + ema) / (w1 + w2 + w3)
