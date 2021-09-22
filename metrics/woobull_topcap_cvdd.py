import re
from typing import List

import pandas as pd
import requests
import seaborn as sns
from matplotlib import pyplot as plt

from globals import HTTP_TIMEOUT
from utils import add_common_markers
from .base_metric import BaseMetric


class WoobullMetric(BaseMetric):
    @property
    def name(self) -> str:
        return 'Woobull'

    @property
    def description(self) -> str:
        return 'Woobull Top Cap - CVDD'

    @staticmethod
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


    def calculate(self, source_df: pd.DataFrame, ax: List[plt.Axes]) -> pd.Series:
        df = source_df.copy()

        response = requests.get('https://charts.woobull.com/bitcoin-price-models/', timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        response_html = response.text

        df_top = self._extract_metric(response_html, 'top_', 'Top')
        df_cvdd = self._extract_metric(response_html, 'cvdd', 'CVDD')
        df_woobull = df_top.merge(df_cvdd, on='Date')

        df = df.join(df_woobull.set_index('Date'), on='Date')

        df['WoobullIndex'] = (df['Price'] - df['CVDD']) / \
                             (df['Top'] - df['CVDD'])

        ax[0].set_title(self.description)
        sns.lineplot(data=df, x='Date', y='WoobullIndex', ax=ax[0])
        add_common_markers(df, ax[0])

        return df['WoobullIndex']
