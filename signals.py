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


def complex_freq(data, w1, w2, w3, d1, d2, d3, sf):
    weights = np.array([w1, w2, w3])
    kernels = np.array([sma_filter(d1), lma_filter(d2), ema_filter(sf, d3)])
    vwma = np.vectorize(lambda kernel: wma(data, kernel))
    return np.dot(weights, vwma(kernels)) / weights.sum()
