import re

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from globals import HTTP_TIMEOUT
from metrics.base_metric import BaseMetric
from utils import HTTP, add_common_markers


def _extract_metric(html: str, html_name: str, df_name: str) -> pd.DataFrame:
    match = re.search(html_name + r"\s*=\s*{\s*x:\s*\[(?P<x>[',\d\s:-]+)],\s*y:\s*\[(?P<y>[,\d.eE-]+)]", html)

    if not match:
        raise Exception(f'Failed to extract the "{html_name}" metric')

    match_x = match.group('x').split(',')
    match_y = match.group('y').split(',')

    df = pd.DataFrame({
        'Date': [x.strip('\'') for x in match_x],
        df_name: [float(y) for y in match_y],
    })
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    return df


def _fetch_df() -> pd.DataFrame:
    response = HTTP.get('https://charts.woobull.com/bitcoin-price-models/')
    response.raise_for_status()
    response_html = response.text

    df_top = _extract_metric(response_html, 'top_', 'Top')
    df_cvdd = _extract_metric(response_html, 'cvdd', 'CVDD')
    df = df_top.merge(df_cvdd, on='Date')

    return df


class WoobullMetric(BaseMetric):
    @property
    def name(self) -> str:
        return 'Woobull'

    @property
    def description(self) -> str:
        return 'Woobull Top Cap vs CVDD'

    def _calculate(self, df: pd.DataFrame, ax: list[plt.Axes]) -> pd.Series:
        df = df.merge(_fetch_df(), on='Date', how='left')
        df['Top'].ffill(inplace=True)
        df['TopLog'] = np.log(df['Top'])
        df['CVDD'].ffill(inplace=True)
        df['CVDDLog'] = np.log(df['CVDD'])

        df['Woobull'] = (df['PriceLog'] - df['CVDDLog']) / \
                        (df['TopLog'] - df['CVDDLog'])

        high_rows = df.loc[df['PriceHigh'] == 1]
        high_x = high_rows.index.values.reshape(-1, 1)
        high_y = high_rows['Woobull'].values.reshape(-1, 1)

        low_rows = df.loc[df['PriceLow'] == 1][1:]
        low_x = low_rows.index.values.reshape(-1, 1)
        low_y = low_rows['Woobull'].values.reshape(-1, 1)

        x = df.index.values.reshape(-1, 1)

        lin_model = LinearRegression()
        lin_model.fit(high_x, high_y)
        df['WoobullHighModel'] = lin_model.predict(x)

        lin_model.fit(low_x, low_y)
        df['WoobullLowModel'] = lin_model.predict(x)

        df['WoobullIndex'] = (df['Woobull'] - df['WoobullLowModel']) / \
                             (df['WoobullHighModel'] - df['WoobullLowModel'])

        ax[0].set_title(self.description)
        sns.lineplot(data=df, x='Date', y='WoobullIndex', ax=ax[0])
        add_common_markers(df, ax[0])

        sns.lineplot(data=df, x='Date', y='Woobull', ax=ax[1])
        sns.lineplot(data=df, x='Date', y='WoobullHighModel', ax=ax[1])
        sns.lineplot(data=df, x='Date', y='WoobullLowModel', ax=ax[1])
        add_common_markers(df, ax[1], price_line=False)

        return df['WoobullIndex']
