from typing import Literal
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd


class KaggleDataset:
    def __init__(self, timescale: Literal["Daily"] | Literal["Hourly"] = "Daily"):
        self.df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "prasoonkottarathil/btcinusd",
            f"BTC-{timescale}.csv",
            pandas_kwargs={
                "usecols": ["date", "close"],
                "parse_dates": ["date"],
                "date_parser": pd.Timestamp,
            },
        )

    # def get_series(self, start: Timestamp)


print(KaggleDataset().df.info())
