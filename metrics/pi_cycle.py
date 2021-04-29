from itertools import zip_longest
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from utils import mark_highs_lows, add_common_markers, split_df_on_index_gap
from .base_metric import BaseMetric


class PiCycleMetric(BaseMetric):
    @property
    def name(self) -> str:
        return 'PiCycle'

    @property
    def description(self) -> str:
        return 'Pi Cycle Top Indicator'

    def calculate(self, source_df: pd.DataFrame, ax: List[plt.Axes]) -> pd.Series:
        df = source_df.copy()

        df['111DMA'] = df['Price'].rolling(111).mean()
        df['350DMAx2'] = df['Price'].rolling(350).mean() * 2

        df['111DMALog'] = np.log(df['111DMA'])
        df['350DMAx2Log'] = np.log(df['350DMAx2'])
        df['PiCycleDiff'] = np.abs(df['111DMALog'] - df['350DMAx2Log'])
        df['PiCycleDiffThreshold'] = 0

        df['PiCycleIsFluke'] = df['111DMA'] > df['350DMAx2']

        df_flukes = [*split_df_on_index_gap(df[df['PiCycleIsFluke']])]
        df_actuals = [*split_df_on_index_gap(df[~df['PiCycleIsFluke']])]

        for df_fluke, df_actual, df_fluke_next in zip_longest(df_flukes, df_actuals[1:], df_flukes[1:], fillvalue=None):
            if df_fluke is None:
                break

            max_divergence_idx = df_fluke['PiCycleDiff'].argmax()
            max_divergence_row = df_fluke.iloc[max_divergence_idx]
            df.loc[max_divergence_row.name < df.index, 'PiCycleDiffThreshold'] = max_divergence_row['PiCycleDiff']

            if df_actual is not None:
                df_actual_above = df_actual[df_actual['PiCycleDiff'] >= max_divergence_row['PiCycleDiff']]

                if df_actual_above.shape[0] > 0:
                    df.loc[df_actual_above.index.min() <= df.index, 'PiCycleDiffThreshold'] = 0

            if df_fluke_next is not None:
                df.loc[df_fluke_next.index.min() <= df.index, 'PiCycleDiffThreshold'] = 0

        df.loc[df['PiCycleDiff'] < df['PiCycleDiffThreshold'], 'PiCycleDiff'] = df['PiCycleDiffThreshold']
        df = mark_highs_lows(df, 'PiCycleDiff', True, round(365 * 2), 365)

        for _, row in df.loc[df['PiCycleDiffHigh'] == 1].iterrows():
            df.loc[df.index > row.name, 'PreviousPiCycleDiffHighValue'] = row['PiCycleDiff']

        df['PiCycleIndex'] = 1 - (df['PiCycleDiff'] / df['PreviousPiCycleDiffHighValue'])
        df.loc[df['PiCycleIndex'] < 0, 'PiCycleIndex'] = 0

        df['PiCycleIndexNoNa'] = df['PiCycleIndex'].fillna(0)
        ax[0].set_title(self.description)
        sns.lineplot(data=df, x='Date', y='PiCycleIndexNoNa', ax=ax[0])
        add_common_markers(df, ax[0])

        return df['PiCycleIndex']
