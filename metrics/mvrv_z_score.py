from typing import List

import pandas as pd
import requests
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from globals import HTTP_TIMEOUT
from utils import add_common_markers, mark_highs_lows
from .base_metric import BaseMetric


def _fetch_df() -> pd.DataFrame:
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

    response = requests.post(
        'https://www.lookintobitcoin.com/django_plotly_dash/app/mvrv_zscore/_dash-update-component', json=request_data,
        timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    response_json = response.json()
    response_x = response_json['response']['props']['figure']['data'][0]['x']
    response_y = response_json['response']['props']['figure']['data'][0]['y']

    df = pd.DataFrame({
        'Date': response_x[:len(response_y)],
        'MVRV': response_y,
    })
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    return df


class MVRVMetric(BaseMetric):
    @property
    def name(self) -> str:
        return 'MVRV'

    @property
    def description(self) -> str:
        return 'MVRV Z-Score'

    def calculate(self, df: pd.DataFrame, ax: List[plt.Axes]) -> pd.Series:
        bull_days_shift = 6

        df = df.merge(_fetch_df(), on='Date', how='left')
        df.loc[df['DaysSinceHalving'] < df['DaysSincePriceLow'], 'MVRV'] = df['MVRV'].shift(bull_days_shift)

        df = mark_highs_lows(df, 'MVRV', True, 120, 120)
        df.fillna({'MVRVHigh': 0, 'MVRVLow': 0}, inplace=True)
        df['MVRV'].ffill(inplace=True)

        df.loc[df['MVRV'] < 6, 'MVRVHigh'] = 0

        high_rows = df.loc[df['MVRVHigh'] == 1]
        high_x = high_rows.index.values.reshape(-1, 1)
        high_y = high_rows['MVRV'].values.reshape(-1, 1)

        low_rows = df.loc[df['PriceLow'] == 1]
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

        sns.lineplot(data=df, x='Date', y='MVRV', ax=ax[1])
        sns.lineplot(data=df, x='Date', y='MVRVHighModel', ax=ax[1])
        sns.lineplot(data=df, x='Date', y='MVRVLowModel', ax=ax[1])
        add_common_markers(df, ax[1], price_line=False)

        return df['MVRVIndex']
