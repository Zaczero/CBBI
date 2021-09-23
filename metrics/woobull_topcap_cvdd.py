import re
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
    response = requests.get('https://charts.woobull.com/bitcoin-price-models/', timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    response_html = response.text

    df_top = _extract_metric(response_html, 'top_', 'Top')
    df_cvdd = _extract_metric(response_html, 'cvdd', 'CVDD')
    df_woobull = df_top.merge(df_cvdd, on='Date')

    return df_woobull


class WoobullMetric(BaseMetric):
    @property
    def name(self) -> str:
        return 'Woobull'

    @property
    def description(self) -> str:
        return 'Woobull Top Cap vs CVDD'

    def calculate(self, source_df: pd.DataFrame, ax: List[plt.Axes]) -> pd.Series:
        df = source_df.copy()
        df = df.merge(_fetch_df(), on='Date', how='left')

        df['Woobull'] = (df['Price'] - df['CVDD']) / \
                        (df['Top'] - df['CVDD'])

        df = mark_highs_lows(df, 'Woobull', False, round(365 * 0.5), 365)
        df.loc[df['Woobull'] < 0.75, 'WoobullHigh'] = 0

        high_rows = df.loc[df['WoobullHigh'] == 1]
        high_x = high_rows.index.values.reshape(-1, 1)
        high_y = high_rows['Woobull'].values.reshape(-1, 1)

        x = df.index.values.reshape(-1, 1)

        lin_model = LinearRegression()
        lin_model.fit(high_x, high_y)
        df['WoobullModelMax'] = lin_model.predict(x)

        df['WoobullModel'] = df['Woobull'] / df['WoobullModelMax']
        df['WoobullModelScaled'] = df['WoobullModel'] * (np.exp(1) - 1) + 1
        df['WoobullIndex'] = np.log(df['WoobullModelScaled'])

        ax[0].set_title(self.description)
        sns.lineplot(data=df, x='Date', y='WoobullIndex', ax=ax[0])
        add_common_markers(df, ax[0])

        return df['WoobullIndex']
