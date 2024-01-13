import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from api.coinsoto_api import cs_fetch
from metrics.base_metric import BaseMetric
from utils import add_common_markers, mark_highs_lows


class MVRVMetric(BaseMetric):
    @property
    def name(self) -> str:
        return 'MVRV'

    @property
    def description(self) -> str:
        return 'MVRV Z-Score'

    def _calculate(self, df: pd.DataFrame, ax: list[plt.Axes]) -> pd.Series:
        bull_days_shift = 6
        low_model_adjust = 0.26

        df = df.merge(
            cs_fetch(
                path='chain/index/charts?type=/charts/mvrv-zscore/',
                data_selector='value4',
                col_name='MVRV',
            ),
            on='Date',
            how='left',
        )
        df.loc[df['DaysSinceHalving'] < df['DaysSincePriceLow'], 'MVRV'] = df['MVRV'].shift(bull_days_shift)
        df['MVRV'].ffill(inplace=True)
        df['MVRV'] = np.log(df['MVRV'] + 1)

        df = mark_highs_lows(df, 'MVRV', True, round(365 * 2), 365)
        df.fillna({'MVRVHigh': 0, 'MVRVLow': 0}, inplace=True)

        high_rows = df.loc[df['MVRVHigh'] == 1]
        high_x = high_rows.index.values.reshape(-1, 1)
        high_y = high_rows['MVRV'].values.reshape(-1, 1)

        low_rows = df.loc[df['PriceLow'] == 1]
        low_x = low_rows.index.values.reshape(-1, 1)
        low_y = low_rows['MVRV'].values.reshape(-1, 1)

        x = df.index.values.reshape(-1, 1)

        lin_model = LinearRegression()
        lin_model.fit(high_x, high_y)
        df['HighModel'] = lin_model.predict(x)

        lin_model.fit(low_x, low_y)
        df['LowModel'] = lin_model.predict(x) + low_model_adjust

        df['Index'] = (df['MVRV'] - df['LowModel']) / (df['HighModel'] - df['LowModel'])

        df['IndexNoNa'] = df['Index'].fillna(0)
        ax[0].set_title(self.description)
        sns.lineplot(data=df, x='Date', y='IndexNoNa', ax=ax[0])
        add_common_markers(df, ax[0])

        sns.lineplot(data=df, x='Date', y='MVRV', ax=ax[1])
        sns.lineplot(data=df, x='Date', y='HighModel', ax=ax[1])
        sns.lineplot(data=df, x='Date', y='LowModel', ax=ax[1])
        add_common_markers(df, ax[1], price_line=False)

        return df['Index']
