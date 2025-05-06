import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datasets.kaggle import KaggleDataset
from signals import sma_filter, lma_filter, ema_filter, wma


def plot_indicators():
    # Load the dataset
    dataset = KaggleDataset(timescale="Daily")
    start_date = pd.Timestamp("2019-01-01")
    duration = pd.Timedelta(days=100)

    # Get the 30-day time series
    series = dataset.get_series(start=start_date, duration=duration)
    prices = series.values
    dates = series.index

    # Define the window size
    N = 10

    # Calculate SMA, LMA, and EMA
    sma = wma(prices, N, sma_filter(N))
    lma = wma(prices, N, lma_filter(N))
    ema = wma(prices, N, ema_filter(0.3, N))  # Example smoothing factor = 0.3

    # Plot the results after the window size
    plt.figure(figsize=(12, 6))
    plt.plot(
        dates,
        prices,
        label="Original Prices",
        color="blue",
        alpha=0.6,
    )
    plt.plot(dates, sma, label="SMA", color="orange")
    plt.plot(dates, lma, label="LMA", color="green")
    plt.plot(dates, ema, label="EMA", color="red")
    plt.title("SMA, LMA, and EMA for 30-Day Window Starting 2019-01-01")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.show()


# Call the function to plot
if __name__ == "__main__":
    plot_indicators()

# import bot
# import pandas as pd
# import test_problem as fitness
# import visual

# close_price = pd.read_csv('training.csv')['close']
# print(close_price)
# x = fitness.bot_fitness_func("sma",10, 20)
# print(x)
# # visual.plot_sma_cross(close_price, short_window=10, long_window=30)
