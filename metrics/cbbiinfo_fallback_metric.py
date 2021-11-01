import traceback
from abc import ABC, abstractmethod
from typing import List

import cli_ui
import pandas as pd
import requests
from matplotlib import pyplot as plt

from globals import HTTP_TIMEOUT
from metrics import BaseMetric


class CBBIInfoFallbackMetric(BaseMetric, ABC):
    def _fetch_df_fallback(self) -> pd.DataFrame:
        response = requests.get(
            'https://colintalkscrypto.com/cbbi/data/latest.json',
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; rv:78.0) Gecko/20100101 Firefox/78.0'},
            timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        response_data = response.json()[self.name]

        df = pd.DataFrame(response_data.items(), columns=[
            'Date',
            'Value',
        ])
        df['Date'] = pd.to_datetime(df['Date'], unit='s').dt.tz_localize(None)

        return df

    def _fallback(self, df: pd.DataFrame) -> pd.Series:
        df = df.merge(self._fetch_df_fallback(), on='Date', how='left')
        df['Value'].ffill(inplace=True)

        return df['Value']

    @abstractmethod
    def _calculate(self, df: pd.DataFrame, ax: List[plt.Axes]) -> pd.Series:
        pass

    def calculate(self, df: pd.DataFrame, ax: List[plt.Axes]) -> pd.Series:
        try:
            return self._calculate(df, ax)

        except Exception:
            traceback.print_exc()
            cli_ui.warning(f'Requesting fallback values for {self.name}')
            return self._fallback(df)
