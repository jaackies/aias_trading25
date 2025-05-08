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
        )[::-1]
        self.df["date"] = pd.to_datetime(self.df["unix"], unit="s")
        self.df = self.df.drop(columns=["unix"])

    def get_timerange(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Get a time range of the dataset. (inclusive start, exclusive end)"""
        return self.df[(self.df["date"] >= start) & (self.df["date"] < end)]
