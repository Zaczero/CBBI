from abc import ABC, abstractmethod

import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from sty import bg, fg, rs

from api.cbbiinfo_api import cbbi_fetch
from utils import add_common_markers, send_error_notification


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
    def _calculate(self, df: pd.DataFrame, ax: list[Axes]) -> pd.Series:
        pass

    def _fallback(self, df: pd.DataFrame, ax: list[Axes]) -> pd.Series:
        df = df.merge(cbbi_fetch(self.name), on='Date', how='left')
        df['Value'] = df['Value'].ffill()

        # Create fallback plots so charts aren't empty
        if ax and len(ax) >= 2:
            # Plot the metric output on the left axis (index 0)
            ax[0].set_title(f"{self.description} (fallback)")
            sns.lineplot(data=df, x='Date', y='Value', ax=ax[0])
            add_common_markers(df, ax[0])
            
            # Plot the same on the right axis (index 1) as we don't have raw data
            sns.lineplot(data=df, x='Date', y='Value', ax=ax[1])
            add_common_markers(df, ax[1], price_line=False)

        return df['Value']

    async def calculate(self, df: pd.DataFrame, ax: list[Axes]) -> pd.Series:
        try:
            return self._calculate(df, ax)
        except Exception as ex:
            # Print full traceback for backend debugging
            import traceback
            traceback.print_exc()
            await send_error_notification(ex)

            print(fg.black + bg.yellow + f' Requesting fallback values for {self.name} (from CBBI.info) ' + rs.all)
            return self._fallback(df, ax)
