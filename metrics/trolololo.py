from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from utils import add_common_markers
from .base_metric import BaseMetric


class TrolololoMetric(BaseMetric):
    @property
    def name(self) -> str:
        return 'Trolololo'

    @property
    def description(self) -> str:
        return 'Bitcoin Trolololo Trend Line'

    def calculate(self, source_df: pd.DataFrame, ax: List[plt.Axes]) -> pd.Series:
        begin_date = pd.to_datetime('2012-01-01')
        log_diff_line_count = 8
        log_diff_bottom = 0.5

        df = source_df.copy()

        df['TroloDaysSinceBegin'] = (df['Date'] - begin_date).dt.days

        df['TroloTopPrice'] = np.power(10, 2.900 * np.log(df['TroloDaysSinceBegin'] + 1400) - 19.463)  # Maximum Bubble Territory
        df['TroloTopPriceLog'] = np.log(df['TroloTopPrice'])
        df['TroloBottomPrice'] = np.power(10, 2.788 * np.log(df['TroloDaysSinceBegin'] + 1200) - 19.463)  # Basically a Fire Sale
        df['TroloBottomPriceLog'] = np.log(df['TroloBottomPrice'])

        df['TroloPriceLogDifference'] = (df['TroloTopPriceLog'] - df['TroloBottomPriceLog']) / log_diff_line_count
        df['TroloBottomPriceLogCorrect'] = df['TroloBottomPriceLog'] - log_diff_bottom * df['TroloPriceLogDifference']
        df['TroloOvershootActual'] = df['PriceLog'] - df['TroloTopPriceLog']

        high_rows = df.loc[(df['PriceHigh'] == 1) & (df['Date'] >= begin_date)]
        high_x = high_rows.index.values.reshape(-1, 1)
        high_y = high_rows['TroloOvershootActual'].values.reshape(-1, 1)
        high_y[0] *= 0.6  # the first value seems too high

        x = df.index.values.reshape(-1, 1)

        lin_model = LinearRegression()
        lin_model.fit(high_x, high_y)
        df['TroloOvershootModel'] = lin_model.predict(x)

        df['TroloIndex'] = (df['PriceLog'] - df['TroloBottomPriceLogCorrect']) / \
                           (df['TroloTopPriceLog'] + df['TroloOvershootModel'] - df['TroloBottomPriceLogCorrect'])

        ax[0].set_title(self.description)
        sns.lineplot(data=df, x='Date', y='TroloIndex', ax=ax[0])
        add_common_markers(df, ax[0])

        return df['TroloIndex']
