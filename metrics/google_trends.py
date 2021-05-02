from datetime import timedelta
from typing import List

import cli_ui
import numpy as np
import pandas as pd
import seaborn as sns
from filecache import filecache
from matplotlib import pyplot as plt
from pytrends.request import TrendReq
from sklearn.linear_model import LinearRegression

from utils import mark_highs_lows, add_common_markers
from .base_metric import BaseMetric


@filecache(3600 * 24 * 3)  # 3 day cache
def fetch_google_trends_data(keyword: str, timeframe: str) -> pd.DataFrame:
    pytrends = TrendReq(retries=5, backoff_factor=1)
    pytrends.build_payload(kw_list=[keyword], timeframe=timeframe)
    return pytrends.interest_over_time()


class GoogleTrendsMetric(BaseMetric):
    @property
    def name(self) -> str:
        return 'GoogleTrends'

    @property
    def description(self) -> str:
        return '"Bitcoin" search term (Google Trends)'

    def calculate(self, source_df: pd.DataFrame, ax: List[plt.Axes]) -> pd.Series:
        drop_off_per_day = 0.012

        df = source_df.copy()

        keyword = 'Bitcoin'
        date_start = df.iloc[0]['Date'] - timedelta(90)
        date_end = df.iloc[-1]['Date']
        date_current = date_start

        delta_days = 269  # 270 days will cause Google Trends API return weekly format
        match_window_days = 200  # int(np.floor(delta_days / 2)) + 1
        iteration_count = int(np.ceil((date_end - date_start) / timedelta(delta_days - match_window_days)))

        df_interest = pd.DataFrame()
        cli_ui.info_1('Google Trends progress: [', end='')

        for i in range(iteration_count):
            if i % 2 == 0:
                print('#', end='')

            date_start_str = date_current.strftime('%Y-%m-%d')
            date_current = min(date_current + timedelta(delta_days), date_end)
            date_end_str = date_current.strftime('%Y-%m-%d')
            date_current -= timedelta(match_window_days - 1)

            timeframe = f'{date_start_str} {date_end_str}'
            df_fetched = fetch_google_trends_data(keyword, timeframe)

            if df_interest.shape[0] > 0:
                prev_scale = np.max(df_interest.iloc[-match_window_days:][keyword])
                next_scale = np.max(df_fetched.iloc[:match_window_days][keyword])
                ratio_scale = next_scale / prev_scale

                if ratio_scale > 1:
                    df_fetched[keyword] /= ratio_scale
                elif ratio_scale < 1:
                    df_interest[keyword] *= ratio_scale

                df_fetched = df_fetched.iloc[match_window_days:]

            df_interest = df_interest.append(df_fetched)

        print(']')

        df_interest.reset_index(inplace=True)
        df_interest.rename(columns={
            'date': 'Date',
            keyword: 'Interest'
        }, inplace=True)
        df_interest = mark_highs_lows(df_interest, 'Interest', False, round(365 * 1.5), 365)

        for _, row in df_interest.loc[df_interest['InterestHigh'] == 1].iterrows():
            from_idx = np.min(df_interest.loc[(df_interest.index > row.name) & (df_interest['Interest'] < 10)].index)
            df_interest.loc[df_interest.index >= from_idx, 'PreviousInterestHigh'] = row['Interest']

        df_interest['InterestScaleActual'] = df_interest['Interest'] / df_interest['PreviousInterestHigh']

        high_rows = df_interest.loc[(df_interest['InterestHigh'] == 1) & ~ (df_interest['PreviousInterestHigh'].isna())]
        high_x = high_rows.index.values.reshape(-1, 1)
        high_y = high_rows['InterestScaleActual'].values.reshape(-1, 1)

        x = df_interest.index.values.reshape(-1, 1)

        lin_model = LinearRegression()
        lin_model.fit(high_x, high_y)
        df_interest['InterestScaleModel'] = lin_model.predict(x)

        df = df.join(df_interest.set_index('Date'), on='Date')
        df.fillna({'InterestHigh': 0, 'InterestLow': 0}, inplace=True)
        df['Interest'].ffill(inplace=True)
        df['PreviousInterestHigh'].ffill(inplace=True)
        df['InterestScaleModel'].ffill(inplace=True)
        df['GoogleTrends'] = df['Interest'] / (df['InterestScaleModel'] * df['PreviousInterestHigh'])

        def calculate_drop_off(rows_ref: np.ndarray):
            rows = np.copy(rows_ref)

            for i, drop_off in zip(range(rows.shape[0] - 1), range(rows.shape[0] - 1, 0, -1)):
                rows[i] -= drop_off * drop_off_per_day

            return np.max(rows)

        df['GoogleTrendsIndex'] = df['GoogleTrends'] \
            .rolling(int(1.2 / drop_off_per_day), min_periods=1) \
            .apply(calculate_drop_off, raw=True)

        ax[0].set_title(self.description)
        sns.lineplot(data=df, x='Date', y='GoogleTrendsIndex', ax=ax[0])
        add_common_markers(df, ax[0])

        return df['GoogleTrendsIndex']
