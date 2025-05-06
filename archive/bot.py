import numpy as np
import pandas as pd

def pad(P, N):
  # where P is your numpy array of data points
  # and N is the window size
  padding = -np.flip(P[1:N])
  return np.append(padding, P)

def sma_filter(N):
  return np.ones(N)/N

def lma_filter(N):
  return (np.full(N, 2)-(np.arange(N)/N))/(N+np.ones(N))
# k is increasing from k=0 to k=n-1 (where n is the number of points in N)

def ema_filter(sf, N):
  # sf = alpha = smoothing factor
  return (np.full(N,sf) * np.power((np.ones(N)-np.full(N,sf)), np.arange(N)))

def wma(P, N, kernel):
  # P = array of data points
  # N = window size
  # kernel = filter method called on N
  return np.convolve(pad(P,N), kernel, "valid")

def complex_freq(data, w1, w2, w3, d1, d2, d3, sf):
  sma = wma(data, d1, sma_filter(d1)) * w1
  lma = wma(data, d2, lma_filter(d2)) * w2
  ema = wma(data, d3, ema_filter(sf, d3)) * w3
  weights = w1 + w2 + w3
  return (sma + lma + ema) / weights

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

def get_signals_sma2(data, highN, lowN):
  # get buy and sell signals by using two SMA filters
  # one with window size highN that is small for high freq signal
  # one with window size lowN that is larger for low freq signal
  high_signal = wma(data, highN, sma_filter(highN))
  low_signal = wma(data, lowN, sma_filter(lowN))
  return buysell_signals(high_signal, low_signal)

def get_signals_smaema(data, EN, lowN, Esf):
  # get buy and sell signals by using a
  # SMA filter for low freq (lowN = larger window size) and
  # EMA for high frequency (EN = smaller window size, Esf = smoothing factor)
  high_signal = wma(data, EN, ema_filter(Esf, EN))
  low_signal = wma(data, lowN, sma_filter(lowN))
  return buysell_signals(high_signal, low_signal)
  
def get_signals_complex(data, high, low):
  # high and low are arrays [w1, w2, w3, d1, d2, d3, sf], defining parameters of
  # weight, length of windows and smoothing factors of the complex frequency
  # high is the array defining the paramters for high frequency
  # low is the array defining the parameteres for low frequency
  high_signal = complex_freq(data, high[0], high[1], high[2], high[3], high[4], high[5], high[6])
  low_signal = complex_freq(data, low[0], low[1], low[2], low[3], low[4], low[5], low[6])
  return buysell_signals(high_signal, low_signal)