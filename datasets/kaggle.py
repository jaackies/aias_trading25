from typing import Literal
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np


class KaggleDataset:
    def __init__(self, timescale: Literal["Daily"] | Literal["Hourly"] = "Daily"):
        self.df: pd.DataFrame = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "prasoonkottarathil/btcinusd",
            f"BTC-{timescale}.csv",
            pandas_kwargs={
                "usecols": ["unix", "close"],
            },
        )[::-1]
        self.df["date"] = pd.to_datetime(self.df["unix"], unit="s")
        self.df = self.df.drop(columns=["unix"])

    def get_price_date_series(
        self, start: pd.Timestamp = None, end: pd.Timestamp = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the price and date series for the given time range. (incl start, excl end)"""
        df_filtered = self.df
        if start is not None:
            df_filtered = df_filtered[df_filtered["date"] >= start]
        if end is not None:
            df_filtered = df_filtered[df_filtered["date"] < end]
        return df_filtered["close"].values, df_filtered["date"].values
