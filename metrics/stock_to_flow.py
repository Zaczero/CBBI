from datetime import timedelta
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from utils import add_common_markers
from .base_metric import BaseMetric


class StockToFlowMetric(BaseMetric):
    @property
    def name(self) -> str:
        return 'StockToFlow'

    @property
    def description(self) -> str:
        return 'Stock-to-Flow Chart'

    def calculate(self, source_df: pd.DataFrame, ax: List[plt.Axes]) -> pd.Series:
        sf_emerge_days = 463
        sf_peak_delay = 60

        df = source_df.copy()

        for _, row in df.loc[df['PriceHigh'] == 1].iterrows():
            df.loc[df.index > row.name, 'PreviousPriceHighDate'] = row['Date']

        for _, row in df.loc[df['Halving'] == 1].iterrows():
            df.loc[df.index > row.name, 'PreviousHalvingDate'] = row['Date']

        df.loc[df['PreviousPriceHighDate'] >= df['PreviousHalvingDate'], 'PreviousHalvingDate'] = df['NextHalvingDate']

        df['StockToFlowTarget'] = df['PreviousHalvingDate'] + timedelta(sf_emerge_days + sf_peak_delay)
        df['StockToFlow'] = (df['Date'] - df['PreviousPriceHighDate']) / \
                            (df['StockToFlowTarget'] - df['PreviousPriceHighDate'])

        df['StockToFlowIndex'] = 1 - np.abs(1 - df['StockToFlow'])
        df.loc[(0 < df['DaysSincePriceHigh']) & (df['DaysSincePriceHigh'] < df['DaysSincePriceLow']), 'StockToFlowIndex'] = np.nan

        df['StockToFlowIndexNoNa'] = df['StockToFlowIndex'].fillna(0)
        ax[0].set_title(self.description)
        sns.lineplot(data=df, x='Date', y='StockToFlowIndexNoNa', ax=ax[0])
        add_common_markers(df, ax[0])

        return df['StockToFlowIndex']
