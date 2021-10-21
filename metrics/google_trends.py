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
from tqdm import tqdm

from utils import mark_highs_lows, add_common_markers
from .base_metric import BaseMetric


@filecache(3600 * 24 * 3)  # cache for 3 days
def _fetch_google_trends_data(keyword: str, timeframe: str) -> pd.DataFrame:
    pytrends = TrendReq(retries=5, backoff_factor=1)
    pytrends.build_payload(kw_list=[keyword], timeframe=timeframe)

    df = pytrends.interest_over_time()
    df.drop(columns=['isPartial'], inplace=True)
    df.reset_index(inplace=True)
    df.rename(columns={
        'date': 'Date',
        keyword: 'Interest'
    }, inplace=True)

    return df


def _normalize(df: pd.DataFrame, df_fetch: pd.DataFrame) -> pd.DataFrame:
    if df.shape[0] == 0:
        return df_fetch

    df_inner = df.merge(df_fetch, how='inner', on='Date')
    overlap_days = df_inner.shape[0]

    prev_scale = np.max(df_inner['Interest_x'])
    next_scale = np.max(df_inner['Interest_y'])
    ratio_scale = next_scale / prev_scale

    if ratio_scale > 1:
        df_fetch['Interest'] /= ratio_scale
    elif ratio_scale < 1:
        df['Interest'] *= ratio_scale

    return df_fetch.iloc[overlap_days:]


def _fetch_df(keyword: str, date_from: pd.Timestamp, date_to: pd.Timestamp) -> pd.DataFrame:
    delta_days = 269  # 270 days will cause Google Trends API return weekly format
    overlap_days = 200

    df = pd.DataFrame()

    date_to -= timedelta(5)  # last 7 days are set by _set_last_week_from_hourly
    date_current = date_from
    iter_count = int(np.ceil((date_to - date_from) / timedelta(delta_days - overlap_days)))

    for _ in tqdm(range(iter_count), desc='Fetching daily Google Trends'):
        timeframe_from = date_current.strftime('%Y-%m-%d')
        date_current = min(date_current + timedelta(delta_days), date_to)
        timeframe_to = date_current.strftime('%Y-%m-%d')
        date_current -= timedelta(overlap_days - 1)

        timeframe = f'{timeframe_from} {timeframe_to}'
        df_fetch = _fetch_google_trends_data(keyword, timeframe)
        df_fetch = _normalize(df, df_fetch)

        df = df.append(df_fetch)

    df = _set_last_week_from_hourly(df, keyword)

    return df


def _set_last_week_from_hourly(df: pd.DataFrame, keyword: str) -> pd.DataFrame:
    df_fetch = _fetch_last_week(keyword)
    df_fetch = df_fetch.resample('1D', on='Date').max()
    df_fetch.drop(columns=['Date'], inplace=True)
    df_fetch.reset_index(inplace=True)
    df_fetch = _normalize(df, df_fetch)

    return df.append(df_fetch)


def _fetch_last_week(keyword: str) -> pd.DataFrame:
    timeframe = f'now 7-d'
    df_fetch = _fetch_google_trends_data(keyword, timeframe)

    return df_fetch


class GoogleTrendsMetric(BaseMetric):
    @property
    def name(self) -> str:
        return 'GoogleTrends'

    @property
    def description(self) -> str:
        return '"Bitcoin" search term (Google Trends)'

    def calculate(self, df: pd.DataFrame, ax: List[plt.Axes]) -> pd.Series:
        keyword = 'Bitcoin'
        days_shift = 1
        drop_off_per_day = 0.012

        df_start_date = df.iloc[0]['Date']
        date_from = df_start_date - timedelta(90)
        date_to = df.iloc[-1]['Date']

        df = df.merge(_fetch_df(keyword, date_from, date_to), on='Date', how='right')
        df['Interest'] = df['Interest'].shift(days_shift, fill_value=np.nan)

        df = mark_highs_lows(df, 'Interest', False, round(365 * 1.5), 365)
        df.fillna({'InterestHigh': 0, 'InterestLow': 0}, inplace=True)

        for _, row in df.loc[df['InterestHigh'] == 1].iterrows():
            from_idx = np.min(df.loc[(df.index > row.name) & (df['Interest'] < row['Interest'] / 3)].index)
            df.loc[df.index >= from_idx, 'PreviousInterest'] = row['Interest']

        df['InterestScale'] = df['Interest'] / df['PreviousInterest']

        high_rows = df.loc[(df['InterestHigh'] == 1) & ~ (df['PreviousInterest'].isna())]
        high_x = high_rows.index.values.reshape(-1, 1)
        high_y = high_rows['InterestScale'].values.reshape(-1, 1)

        x = df.index.values.reshape(-1, 1)

        lin_model = LinearRegression()
        lin_model.fit(high_x, high_y)
        df['InterestScaleModel'] = lin_model.predict(x)

        df = df.loc[df['Date'] >= df_start_date]
        df.reset_index(inplace=True)

        df['GoogleTrends'] = df['Interest'] / (df['InterestScaleModel'] * df['PreviousInterest'])

        def calculate_drop_off(rows_ref: np.ndarray):
            rows = np.copy(rows_ref)

            for i, drop_off in enumerate(range(rows.shape[0] - 1, 0, -1)):
                rows[i] -= drop_off * drop_off_per_day

            return np.max(rows)

        df['GoogleTrendsDropOff'] = df['GoogleTrends'] \
            .rolling(int(1.2 / drop_off_per_day), min_periods=1) \
            .apply(calculate_drop_off, raw=True)

        df['GoogleTrendsDropOffLog'] = np.log(df['GoogleTrendsDropOff'] * 100 + 1)
        df['GoogleTrendsIndex'] = np.interp(df['GoogleTrendsDropOffLog'],
                                            (df['GoogleTrendsDropOffLog'].min(), df['GoogleTrendsDropOffLog'].max()),
                                            (0, 1))

        ax[0].set_title(self.description)
        sns.lineplot(data=df, x='Date', y='GoogleTrendsIndex', ax=ax[0])
        add_common_markers(df, ax[0])

        sns.lineplot(data=df, x='Date', y='Interest', ax=ax[1])
        sns.lineplot(data=df, x='Date', y='PreviousInterest', ax=ax[1])
        sns.lineplot(data=df, x='Date', y='InterestScale', ax=ax[1])
        sns.lineplot(data=df, x='Date', y='InterestScaleModel', ax=ax[1])
        add_common_markers(df, ax[1], price_line=False)

        return df['GoogleTrendsIndex']
