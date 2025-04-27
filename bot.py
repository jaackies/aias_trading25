import numpy as np
import pandas as pd

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = ""

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "prasoonkottarathil/btcinusd",
  file_path,
)

def pad(P, N):
  # where P is your array of data points
  # and N is the # data pt.
  padding = -np.flip(P[1:N])
  return np.append(padding, P)

def sma_filter(N):
  return np.ones(N)/N

def lma_filter(N):
  return (np.full(N, 2)-(np.arange(N.size)/N))/(N+np.ones(N))
# k is increasing from k=0 to k=n-1 (where n is the number of points in N)

def ema_filter(sf, N):
  # sf = alpha = smoothing factor
  return (np.full(N, sf) * (np.ones(N)-np.full(N, sf))^np.arange(N.size))

def wma(P, N, kernel):
  # P = array of data points
  # N = number of days
  # kernel = filter method called on N
  return np.convolve(pad(P,N), kernel, "valid")


# There are two most common approaches to choosing the two signals.
# 
# In both you choose two signals/moving averages, one to be your responsive signal,
# and one to be your smoothed signal
# 
# The first is to choose two moving averages of the same type
# but different durations (or window sizes N), with the smaller duration being
# the more responsive.
# We have already seen from Figure 3, for example, that a shorter term
# (smaller N) SMA reacts more quickly to higher frequency changes in the input than
# a longer term SMA.
# 
# The second is to choose a different type of WMA that is more responsive. 
# The most common would be to choose an EMA for the more responsive signal,
# and an SMA for the smoothed trend signal.
# We have also seen an example of this in Figure 5. In both cases we can see
# intuitively that the crossover points would seem to be reasonable places to buy and sell.

def compound(k)