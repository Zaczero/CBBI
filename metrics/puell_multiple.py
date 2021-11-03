from typing import List

import numpy as np
import pandas as pd
import requests
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from globals import HTTP_TIMEOUT
from utils import add_common_markers
from . import CBBIInfoFallbackMetric


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
                'value': '/charts/puell_multiple/'
            }
        ]
    }

    response = requests.post(
        'https://www.lookintobitcoin.com/django_plotly_dash/app/puell_multiple/_dash-update-component',
        json=request_data,
        timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    response_json = response.json()
    response_x = response_json['response']['props']['figure']['data'][1]['x']
    response_y = response_json['response']['props']['figure']['data'][1]['y']

    df = pd.DataFrame({
        'Date': response_x[:len(response_y)],
        'Puell': response_y,
    })
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    return df


class PuellMetric(CBBIInfoFallbackMetric):
    @property
    def name(self) -> str:
        return 'Puell'

    @property
    def description(self) -> str:
        return 'Puell Multiple'

    def _calculate(self, df: pd.DataFrame, ax: List[plt.Axes]) -> pd.Series:
        df = df.merge(_fetch_df(), on='Date', how='left')
        df['Puell'].ffill(inplace=True)
        df['PuellLog'] = np.log(df['Puell'])

        high_rows = df.loc[df['PriceHigh'] == 1]
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
