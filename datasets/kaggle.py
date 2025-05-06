from typing import Literal
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd


class KaggleDataset:
    def __init__(self, timescale: Literal["Daily"] | Literal["Hourly"] = "Daily"):
        self.df: pd.DataFrame = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "prasoonkottarathil/btcinusd",
            f"BTC-{timescale}.csv",
            pandas_kwargs={
                "usecols": ["unix", "close"],
            },
        )
        self.df["date"] = pd.to_datetime(self.df["unix"], unit="s")
        self.df = self.df.drop(columns=["unix"])

    def get_series(self, start: pd.Timestamp, duration: pd.Timedelta) -> pd.Series:
        """
        Get a time series from the dataset.
        :param start: Start date of the time series.
        :param duration: Duration of the time series.
        :return: Time series as a pandas Series.
        """
        end = start + duration
        return self.df[(self.df["date"] >= start) & (self.df["date"] < end)]["close"]
