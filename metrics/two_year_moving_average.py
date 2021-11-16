from typing import List

import numpy as np
import pandas as pd
import requests
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from globals import HTTP_TIMEOUT
from metrics import BaseMetric
from utils import add_common_markers


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
                'value': '/charts/bitcoin-investor-tool/'
            }
        ]
    }

    response = requests.post(
        'https://www.lookintobitcoin.com/django_plotly_dash/app/market_cycle_ma/_dash-update-component',
        json=request_data,
        timeout=HTTP_TIMEOUT
    )

    response.raise_for_status()
    response_json = response.json()
    response_x = response_json['response']['props']['figure']['data'][2]['x']
    response_y = response_json['response']['props']['figure']['data'][2]['y']

    df = pd.DataFrame({
        'Date': response_x[:len(response_y)],
        '2YMA': response_y,
    })
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    return df


class TwoYearMovingAverageMetric(BaseMetric):
    @property
    def name(self) -> str:
        return '2YMA'

    @property
    def description(self) -> str:
        return '2 Year Moving Average'

    def _calculate(self, df: pd.DataFrame, ax: List[plt.Axes]) -> pd.Series:
        df = df.merge(_fetch_df(), on='Date', how='left')
        df['2YMA'].ffill(inplace=True)
        df['2YMALog'] = np.log(df['2YMA'])
        df['2YMALogDiff'] = df['PriceLog'] - df['2YMALog']

        high_rows = df.loc[df['PriceHigh'] == 1]
        high_x = high_rows.index.values.reshape(-1, 1)
        high_y = high_rows['2YMALogDiff'].values.reshape(-1, 1)

        low_rows = df.loc[df['PriceLow'] == 1]
        low_x = low_rows.index.values.reshape(-1, 1)
        low_y = low_rows['2YMALogDiff'].values.reshape(-1, 1)

        x = df.index.values.reshape(-1, 1)

        lin_model = LinearRegression()
        lin_model.fit(high_x, high_y)
        df['2YMALogOvershootModel'] = lin_model.predict(x)

        lin_model.fit(low_x, low_y)
        df['2YMALogUndershootModel'] = lin_model.predict(x)

        df['2YMAHighModel'] = df['2YMALogOvershootModel'] + df['2YMALog']
        df['2YMALowModel'] = df['2YMALogUndershootModel'] + df['2YMALog']

        df['2YMAIndex'] = (df['PriceLog'] - df['2YMALowModel']) / \
                          (df['2YMAHighModel'] - df['2YMALowModel'])

        df['2YMAIndexNoNa'] = df['2YMAIndex'].fillna(0)
        ax[0].set_title(self.description)
        sns.lineplot(data=df, x='Date', y='2YMAIndexNoNa', ax=ax[0])
        add_common_markers(df, ax[0])

        sns.lineplot(data=df, x='Date', y='PriceLog', ax=ax[1])
        sns.lineplot(data=df, x='Date', y='2YMAHighModel', ax=ax[1])
        sns.lineplot(data=df, x='Date', y='2YMALowModel', ax=ax[1])
        add_common_markers(df, ax[1], price_line=False)

        return df['2YMAIndex']
