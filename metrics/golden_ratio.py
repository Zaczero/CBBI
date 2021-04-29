from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from utils import add_common_markers
from .base_metric import BaseMetric


class GoldenRatioMetric(BaseMetric):
    @property
    def name(self) -> str:
        return 'GoldenRatio'

    @property
    def description(self) -> str:
        return 'The Golden 50.8%-49.2% Ratio'

    def calculate(self, source_df: pd.DataFrame, ax: List[plt.Axes]) -> pd.Series:
        df = source_df.copy()

        df['DaysBetweenPriceLowAndLastHalving'] = df['DaysSincePriceLow'] - df['DaysSinceHalving']
        df['DaysBetweenPriceLowAndNextHalving'] = df['DaysSincePriceLow'] + df['DaysToHalving'].dt.days
        df['DaysBetweenPriceLowAndHalving'] = df['DaysBetweenPriceLowAndLastHalving'].where(df['DaysBetweenPriceLowAndLastHalving'] >= 0, df['DaysBetweenPriceLowAndNextHalving'])

        df['GoldenRatioProjected'] = df['DaysBetweenPriceLowAndHalving'] / 0.506
        df['GoldenRatio'] = df['DaysSincePriceLow'] / df['GoldenRatioProjected']
        df['GoldenRatioIndex'] = 1 - np.abs(1 - df['GoldenRatio'])
        df.loc[(0 < df['DaysSincePriceHigh']) & (df['DaysSincePriceHigh'] < df['DaysSincePriceLow']), 'GoldenRatioIndex'] = np.nan

        df['GoldenRatioIndexNoNa'] = df['GoldenRatioIndex'].fillna(0)
        ax[0].set_title(self.description)
        sns.lineplot(data=df, x='Date', y='GoldenRatioIndexNoNa', ax=ax[0])
        add_common_markers(df, ax[0])

        return df['GoldenRatioIndex']
