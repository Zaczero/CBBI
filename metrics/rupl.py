from typing import List

import pandas as pd
import requests
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from globals import HTTP_TIMEOUT
from utils import mark_highs_lows, add_common_markers
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
                'value': '/charts/relative-unrealized-profit--loss/'
            }
        ]
    }

    response = requests.post(
        'https://www.lookintobitcoin.com/django_plotly_dash/app/unrealised_profit_loss/_dash-update-component',
        json=request_data,
        timeout=HTTP_TIMEOUT
    )

    response.raise_for_status()
    response_json = response.json()
    response_x = response_json['response']['props']['figure']['data'][0]['x']
    response_y = response_json['response']['props']['figure']['data'][0]['y']

    df = pd.DataFrame({
        'Date': response_x[:len(response_y)],
        'RUPL': response_y,
    })
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    return df


class RUPLMetric(BaseMetric):
    @property
    def name(self) -> str:
        return 'RUPL'

    @property
    def description(self) -> str:
        return 'RUPL/NUPL Chart'

    def calculate(self, df: pd.DataFrame, ax: List[plt.Axes]) -> pd.Series:
        df = df.merge(_fetch_df(), on='Date', how='left')

        df = mark_highs_lows(df, 'RUPL', True, round(365 * 2), 365)
        df.fillna({'RUPLHigh': 0, 'RUPLLow': 0}, inplace=True)
        df['RUPL'].ffill(inplace=True)

        high_rows = df.loc[df['RUPLHigh'] == 1]
        high_x = high_rows.index.values.reshape(-1, 1)
        high_y = high_rows['RUPL'].values.reshape(-1, 1)

        low_rows = df.loc[df['RUPLLow'] == 1][1:]
        low_x = low_rows.index.values.reshape(-1, 1)
        low_y = low_rows['RUPL'].values.reshape(-1, 1)

        x = df.index.values.reshape(-1, 1)

        lin_model = LinearRegression()
        lin_model.fit(high_x, high_y)
        df['RUPLHighModel'] = lin_model.predict(x)

        lin_model.fit(low_x, low_y)
        df['RUPLLowModel'] = lin_model.predict(x)

        df['RUPLIndex'] = (df['RUPL'] - df['RUPLLowModel']) / \
                          (df['RUPLHighModel'] - df['RUPLLowModel'])

        ax[0].set_title(self.description)
        sns.lineplot(data=df, x='Date', y='RUPLIndex', ax=ax[0])
        add_common_markers(df, ax[0])

        return df['RUPLIndex']
