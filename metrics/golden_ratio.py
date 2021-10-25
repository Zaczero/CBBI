from datetime import timedelta
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

    def calculate(self, df: pd.DataFrame, ax: List[plt.Axes]) -> pd.Series:
        peak_decline_after_days = 30
        peak_decline_duration = 270

        df['DaysBetweenPriceLowAndLastHalving'] = df['DaysSincePriceLow'] - df['DaysSinceHalving']
        df['DaysBetweenPriceLowAndNextHalving'] = df['DaysSincePriceLow'] + df['DaysToHalving'].dt.days
        df['DaysBetweenPriceLowAndHalving'] = df['DaysBetweenPriceLowAndLastHalving'].where(df['DaysBetweenPriceLowAndLastHalving'] >= 0, df['DaysBetweenPriceLowAndNextHalving'])

        df['GoldenRatioProjected'] = df['DaysBetweenPriceLowAndHalving'] / 0.506
        df['GoldenRatio'] = df['DaysSincePriceLow'] / df['GoldenRatioProjected']
        df['GoldenRatioIndex'] = np.fmin(df['GoldenRatio'], 1, where=df['GoldenRatio'].notna())

        df.loc[(peak_decline_after_days < df['DaysSincePriceHigh']) & (df['DaysSincePriceHigh'] < df['DaysSincePriceLow']), 'GoldenRatioIndex'] = 1 - (df['DaysSincePriceHigh'] - peak_decline_after_days) / peak_decline_duration
        df.loc[(peak_decline_after_days + peak_decline_duration < df['DaysSincePriceHigh']) & (df['DaysSincePriceHigh'] < df['DaysSincePriceLow']), 'GoldenRatioIndex'] = 0

        # force decline until the metric averages out with the CBBI score to be later removed
        force_decline_start = pd.to_datetime('2021-10-24')
        force_decline_duration = timedelta(days=30)
        df.loc[force_decline_start <= df['Date'], 'GoldenRatioIndex'] = 1 - (df['Date'] - force_decline_start) / force_decline_duration

        df['GoldenRatioIndexNoNa'] = df['GoldenRatioIndex'].fillna(0)
        ax[0].set_title(self.description)
        sns.lineplot(data=df, x='Date', y='GoldenRatioIndexNoNa', ax=ax[0])
        add_common_markers(df, ax[0])

        return df['GoldenRatioIndex']
