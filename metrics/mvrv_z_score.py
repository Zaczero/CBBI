from typing import List

import pandas as pd
import requests
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from globals import HTTP_TIMEOUT
from utils import add_common_markers, mark_highs_lows
from .base_metric import BaseMetric


class MVRVMetric(BaseMetric):
    @property
    def name(self) -> str:
        return 'MVRV'

    @property
    def description(self) -> str:
        return 'MVRV Z-Score'

    def calculate(self, df: pd.DataFrame, ax: List[plt.Axes]) -> pd.Series:
        bull_days_shift = 6

        request_data = {
            'output': 'chart.figure',
            'changedPropIds': [
                'url.pathname'
            ],
            'inputs': [
                {
                    'id': 'url',
                    'property': 'pathname',
                    'value': '/charts/mvrv-zscore/'
                }
            ]
        }

        response = requests.post('https://www.lookintobitcoin.com/django_plotly_dash/app/mvrv_zscore/_dash-update-component', json=request_data, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        response_json = response.json()
        response_x = response_json['response']['props']['figure']['data'][0]['x']
        response_y = response_json['response']['props']['figure']['data'][0]['y']

        df_mvrv = pd.DataFrame({
            'Date': response_x[:len(response_y)],
            'MVRV': response_y,
        })
        df_mvrv['Date'] = pd.to_datetime(df_mvrv['Date']).dt.tz_localize(None)

        df = df.join(df_mvrv.set_index('Date'), on='Date')
        df.loc[df['DaysSinceHalving'] < df['DaysSincePriceLow'], 'MVRV'] = df['MVRV'].shift(bull_days_shift)
        df = mark_highs_lows(df, 'MVRV', True, round(365 * 2), 365)
        df.fillna({'MVRVHigh': 0, 'MVRVLow': 0}, inplace=True)
        df['MVRV'].ffill(inplace=True)

        high_rows = df.loc[df['PriceHigh'] == 1]
        high_x = high_rows.index.values.reshape(-1, 1)
        high_y = high_rows['MVRV'].values.reshape(-1, 1)

        low_rows = df.loc[df['MVRVLow'] == 1]
        low_x = low_rows.index.values.reshape(-1, 1)
        low_y = low_rows['MVRV'].values.reshape(-1, 1)

        x = df.index.values.reshape(-1, 1)

        lin_model = LinearRegression()
        lin_model.fit(high_x, high_y)
        df['MVRVHighModel'] = lin_model.predict(x)

        lin_model.fit(low_x, low_y)
        df['MVRVLowModel'] = lin_model.predict(x)

        df['MVRVIndex'] = (df['MVRV'] - df['MVRVLowModel']) / \
                          (df['MVRVHighModel'] - df['MVRVLowModel'])

        df['MVRVIndexNoNa'] = df['MVRVIndex'].fillna(0)
        ax[0].set_title(self.description)
        sns.lineplot(data=df, x='Date', y='MVRVIndexNoNa', ax=ax[0])
        add_common_markers(df, ax[0])

        return df['MVRVIndex']
