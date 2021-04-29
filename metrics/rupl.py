from typing import List

import pandas as pd
import requests
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from globals import HTTP_TIMEOUT
from utils import mark_highs_lows, add_common_markers
from .base_metric import BaseMetric


class RUPLMetric(BaseMetric):
    @property
    def name(self) -> str:
        return 'RUPL'

    @property
    def description(self) -> str:
        return 'RUPL/NUPL Chart'

    def calculate(self, source_df: pd.DataFrame, ax: List[plt.Axes]) -> pd.Series:
        df = source_df.copy()

        response = requests.get('https://www.lookintobitcoin.com/django_plotly_dash/app/unrealised_profit_loss/_dash-layout', timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        response_json = response.json()
        response_x = response_json['props']['children'][0]['props']['figure']['data'][0]['x']
        response_y = response_json['props']['children'][0]['props']['figure']['data'][0]['y']

        df_rupl = pd.DataFrame({
            'Date': response_x[:len(response_y)],
            'RUPL': response_y,
        })
        df_rupl['Date'] = pd.to_datetime(df_rupl['Date']).dt.tz_localize(None)
        df_rupl = mark_highs_lows(df_rupl, 'RUPL', True, round(365 * 2), 365)

        df = df.join(df_rupl.set_index('Date'), on='Date')
        df.fillna({'RUPLHigh': 0, 'RUPLLow': 0}, inplace=True)
        df['RUPL'].ffill(inplace=True)

        low_rows = df.loc[df['RUPLLow'] == 1][1:]
        low_x = low_rows.index.values.reshape(-1, 1)
        low_y = low_rows['RUPL'].values.reshape(-1, 1)

        high_rows = df.loc[df['RUPLHigh'] == 1]
        high_x = high_rows.index.values.reshape(-1, 1)
        high_y = high_rows['RUPL'].values.reshape(-1, 1)

        x = df.index.values.reshape(-1, 1)

        lin_model = LinearRegression()
        lin_model.fit(low_x, low_y)
        df['RUPLLowModel'] = lin_model.predict(x)

        lin_model.fit(high_x, high_y)
        df['RUPLHighModel'] = lin_model.predict(x)

        df['RUPLIndex'] = (df['RUPL'] - df['RUPLLowModel']) / \
                          (df['RUPLHighModel'] - df['RUPLLowModel'])

        ax[0].set_title(self.description)
        sns.lineplot(data=df, x='Date', y='RUPLIndex', ax=ax[0])
        add_common_markers(df, ax[0])

        return df['RUPLIndex']
