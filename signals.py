import numpy as np


def pad(P, N):
    # where P is your array of data points
    # and N is the # data pt.
    padding = -np.flip(P[1:N])
    return np.append(padding, P)


def sma_filter(N):
    return np.ones(N) / N


def lma_filter(N):
    return (np.full(N, 2) / (N + 1)) * (1 - (np.arange(N) / N))


def ema_filter(sf, N):
    w = sf * ((1 - sf) ** np.arange(N))
    return w / np.sum(w)  # Normalize the weights


def wma(P, N, kernel):
    # P = array of data points
    # N = number of days
    # kernel = filter method called on N
    return np.convolve(pad(P, N), kernel, "valid")
