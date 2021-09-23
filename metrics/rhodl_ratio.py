from typing import List

import numpy as np
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
                'value': '/charts/rhodl-ratio/'
            }
        ]
    }

    response = requests.post(
        'https://www.lookintobitcoin.com/django_plotly_dash/app/rhodl_ratio/_dash-update-component',
        json=request_data,
        timeout=HTTP_TIMEOUT
    )

    response.raise_for_status()
    response_json = response.json()
    response_x = response_json['response']['props']['figure']['data'][1]['x']
    response_y = response_json['response']['props']['figure']['data'][1]['y']

    df = pd.DataFrame({
        'Date': response_x[:len(response_y)],
        'RHODL': response_y,
    })
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    return df


class RHODLMetric(BaseMetric):
    @property
    def name(self) -> str:
        return 'RHODL'

    @property
    def description(self) -> str:
        return 'RHODL Ratio'

    def calculate(self, df: pd.DataFrame, ax: List[plt.Axes]) -> pd.Series:
        df = df.merge(_fetch_df(), on='Date', how='left')

        df = mark_highs_lows(df, 'RHODL', True, round(365 * 2), 365)
        df.fillna({'RHODLHigh': 0, 'RHODLLow': 0}, inplace=True)
        df['RHODL'].ffill(inplace=True)
        df['RHODLLog'] = np.log(df['RHODL'])

        high_rows = df.loc[df['RHODLHigh'] == 1]
        high_x = high_rows.index.values.reshape(-1, 1)
        high_y = high_rows['RHODLLog'].values.reshape(-1, 1)

        low_rows = df.loc[df['RHODLLow'] == 1][1:]
        low_x = low_rows.index.values.reshape(-1, 1)
        low_y = low_rows['RHODLLog'].values.reshape(-1, 1)

        x = df.index.values.reshape(-1, 1)

        lin_model = LinearRegression()
        lin_model.fit(high_x, high_y)
        df['RHODLLogHighModel'] = lin_model.predict(x)

        lin_model.fit(low_x, low_y)
        df['RHODLLogLowModel'] = lin_model.predict(x)

        df['RHODLIndex'] = (df['RHODLLog'] - df['RHODLLogLowModel']) / \
                           (df['RHODLLogHighModel'] - df['RHODLLogLowModel'])

        df['RHODLIndexNoNa'] = df['RHODLIndex'].fillna(0)
        ax[0].set_title(self.description)
        sns.lineplot(data=df, x='Date', y='RHODLIndexNoNa', ax=ax[0])
        add_common_markers(df, ax[0])

        return df['RHODLIndex']
