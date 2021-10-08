from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from utils import add_common_markers, mark_highs_lows
from .base_metric import BaseMetric


class PuellMetric(BaseMetric):
    @property
    def name(self) -> str:
        return 'Puell'

    @property
    def description(self) -> str:
        return 'Puell Multiple'

    def calculate(self, df: pd.DataFrame, ax: List[plt.Axes]) -> pd.Series:
        df['PuellMA365'] = df['TotalGenerationUSD'].rolling(365).mean()
        df['Puell'] = df['TotalGenerationUSD'] / df['PuellMA365']
        df['PuellLog'] = np.log(df['Puell'])

        df = mark_highs_lows(df, 'PuellLog', True, 120, 120)
        df.fillna({'PuellLogHigh': 0, 'PuellLogLow': 0}, inplace=True)
        df.loc[df['PuellLog'] < 1.1, 'PuellLogHigh'] = 0

        high_rows = df.loc[df['PuellLogHigh'] == 1]
        high_x = high_rows.index.values.reshape(-1, 1)
        high_y = high_rows['PuellLog'].values.reshape(-1, 1)

        low_rows = df.loc[df['PriceLow'] == 1][1:]
        low_x = low_rows.index.values.reshape(-1, 1)
        low_y = low_rows['PuellLog'].values.reshape(-1, 1)

        x = df.index.values.reshape(-1, 1)

        lin_model = LinearRegression()
        lin_model.fit(high_x, high_y)
        df['PuellLogHighModel'] = lin_model.predict(x)

        lin_model.fit(low_x, low_y)
        df['PuellLogLowModel'] = lin_model.predict(x)

        df['PuellIndex'] = (df['PuellLog'] - df['PuellLogLowModel']) / \
                           (df['PuellLogHighModel'] - df['PuellLogLowModel'])

        df['PuellIndexNoNa'] = df['PuellIndex'].fillna(0)
        ax[0].set_title(self.description)
        sns.lineplot(data=df, x='Date', y='PuellIndexNoNa', ax=ax[0])
        add_common_markers(df, ax[0])

        sns.lineplot(data=df, x='Date', y='PuellLog', ax=ax[1])
        sns.lineplot(data=df, x='Date', y='PuellLogHighModel', ax=ax[1])
        sns.lineplot(data=df, x='Date', y='PuellLogLowModel', ax=ax[1])
        add_common_markers(df, ax[1], price_line=False)

        return df['PuellIndex']
