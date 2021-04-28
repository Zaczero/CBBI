from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from utils import mark_highs_lows, add_common_markers
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

        df = mark_highs_lows(df, 'PiCycleDiff', True, round(365 * 2), 365)

        for _, row in df.loc[df['PiCycleDiffHigh'] == 1].iterrows():
            df.loc[df.index > row.name, 'PreviousPiCycleDiffHighValue'] = row['PiCycleDiff']

        df['PiCycleIndex'] = 1 - (df['PiCycleDiff'] / df['PreviousPiCycleDiffHighValue'])

        df['PiCycleIndexNoNa'] = df['PiCycleIndex'].fillna(0)
        ax[0].set_title(self.description)
        sns.lineplot(data=df, x='Date', y='PiCycleIndexNoNa', ax=ax[0])
        add_common_markers(df, ax[0])

        return df['PiCycleIndex']
