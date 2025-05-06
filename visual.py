import numpy as np
import matplotlib.pyplot as plt

def plot_sma_cross(price, short_window=10, long_window=30):
    short_sma = np.convolve(price, np.ones(short_window)/short_window, mode='valid')
    long_sma = np.convolve(price, np.ones(long_window)/long_window, mode='valid')

    # 让两个 SMA 对齐长度（截取相同部分）
    min_len = min(len(short_sma), len(long_sma))
    short_sma = short_sma[-min_len:]
    long_sma = long_sma[-min_len:]
    price_cut = price[-min_len:]

    # 差值判断买卖
    difference = short_sma - long_sma
    signals = np.full(min_len, "none", dtype=object)

    for i in range(1, min_len):
        if difference[i-1] < 0 and difference[i] > 0:
            signals[i] = "buy"
        elif difference[i-1] > 0 and difference[i] < 0:
            signals[i] = "sell"

    # 可视化
    plt.figure(figsize=(12,6))
    plt.plot(price_cut, label="Price", color='gray', alpha=0.4)
    plt.plot(short_sma, label=f"SMA Short ({short_window})", color='blue')
    plt.plot(long_sma, label=f"SMA Long ({long_window})", color='orange')

    # 标记买卖点
    for i in range(1, min_len):
        if signals[i] == "buy":
            plt.scatter(i, price_cut[i], marker='^', color='green', label='Buy' if i == 1 else "")
        elif signals[i] == "sell":
            plt.scatter(i, price_cut[i], marker='v', color='red', label='Sell' if i == 1 else "")

    plt.title("SMA Crossover Buy/Sell Signal")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
