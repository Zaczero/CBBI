import traceback
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
from sty import fg, bg, rs

from api.cbbiinfo_api import cbbi_fetch


class BaseMetric(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def _calculate(self, df: pd.DataFrame, ax: list[plt.Axes]) -> pd.Series:
        pass

    def _fallback(self, df: pd.DataFrame) -> pd.Series:
        df = df.merge(cbbi_fetch(self.name), on='Date', how='left')
        df['Value'].ffill(inplace=True)

        return df['Value']

    def calculate(self, df: pd.DataFrame, ax: list[plt.Axes]) -> pd.Series:
        try:
            return self._calculate(df, ax)
        except Exception:
            traceback.print_exc()
            print(fg.black + bg.yellow + f' Requesting fallback values for {self.name} (from CBBI.info) ' + rs.all)
            return self._fallback(df)
