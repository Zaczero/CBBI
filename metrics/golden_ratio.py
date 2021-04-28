import numpy as np
import pandas as pd

from .base_metric import BaseMetric


class GoldenRatioMetric(BaseMetric):
    @property
    def name(self) -> str:
        return 'GoldenRatio'

    @property
    def description(self) -> str:
        return 'The Golden 51%-49% Ratio'

    def calculate(self, source_df: pd.DataFrame) -> pd.Series:
        df = source_df.copy()

        df['DaysBetweenPriceLowAndHalving'] = df['DaysSincePriceLow'] - df['DaysSinceHalving']
        df['GoldenRatioProjected'] = df['DaysBetweenPriceLowAndHalving'] / 0.51
        df['GoldenRatio'] = df['DaysSincePriceLow'] / df['GoldenRatioProjected']

        df['GoldenRatioIndex'] = 1 - np.abs(1 - df['GoldenRatio'])
        df.loc[df['DaysBetweenPriceLowAndHalving'] < 0, 'GoldenRatioIndex'] = np.nan
        return df['GoldenRatioIndex']
