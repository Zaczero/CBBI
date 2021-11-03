import os
from datetime import timedelta
from typing import List

import filecache
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pytrends.request import TrendReq
from tqdm import tqdm

from metrics import CBBIInfoFallbackMetric
from utils import mark_highs_lows, add_common_markers

pytrends: TrendReq


@filecache.filecache(3 * filecache.DAY)
def _fetch_google_trends(keyword: str, timeframe: str) -> pd.DataFrame:
    return _fetch_google_trends_nocache(keyword, timeframe)


def _fetch_google_trends_nocache(keyword: str, timeframe: str) -> pd.DataFrame:
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
        df_fetch = _fetch_google_trends(keyword, timeframe)
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
    df_fetch = _fetch_google_trends_nocache(keyword, timeframe)

    return df_fetch


class GoogleTrendsMetric(CBBIInfoFallbackMetric):
    @property
    def name(self) -> str:
        return 'GoogleTrends'

    @property
    def description(self) -> str:
        return '"Bitcoin" search term (Google Trends)'

    _log_intensity = 2
    _hybrid_separator = 0.85
    _hybrid_scale_target = 3

    def __init__(self):
        global pytrends

        proxy = os.environ.get('GOOGLE_PROXY')
        proxies = [proxy] if proxy else []

        pytrends = TrendReq(retries=5, backoff_factor=1, proxies=proxies)

    def _calculate_for_accuracy(self, df: pd.DataFrame) -> pd.Series:
        series = np.interp(df['InterestScale'],
                           (df['InterestScale'].min(), self._hybrid_separator),
                           (0, 1))

        scaled = np.log(series * self._log_intensity + 1) / \
                 np.log(self._log_intensity + 1)

        return np.interp(scaled, (0, 1), (0, self._hybrid_separator))

    def _calculate_for_peaks(self, df: pd.DataFrame) -> pd.Series:
        series = (df['InterestScale'] - self._hybrid_separator) / \
                 (self._hybrid_scale_target - self._hybrid_separator)

        scaled = np.log(series * self._log_intensity + 1) / \
                 np.log(self._log_intensity + 1)

        return np.interp(scaled, (0, 1), (self._hybrid_separator, 1))

    def _calculate(self, df: pd.DataFrame, ax: List[plt.Axes]) -> pd.Series:
        keyword = 'Bitcoin'
        days_shift = ma_days = 5
        max_change_skip_head = 1000
        max_change_interest_valid = 5  # 500%

        date_start = df.iloc[0]['Date']
        date_start_fetch = date_start - timedelta(90)
        date_end = df.iloc[-1]['Date']

        df = df.merge(_fetch_df(keyword, date_start_fetch, date_end), on='Date', how='outer', sort=True)

        max_change_interest = df['Interest'].pct_change()[max_change_skip_head:].max()
        if max_change_interest > max_change_interest_valid:
            raise Exception(f'Interest change is too high: {max_change_interest:%}')

        df['Interest'] = df['Interest'].shift(days_shift, fill_value=np.nan).rolling(ma_days).mean()
        df['Interest'].ffill(inplace=True)

        df = mark_highs_lows(df, 'Interest', False, round(365 * 1.5), 365)
        df.fillna({'InterestHigh': 0, 'InterestLow': 0}, inplace=True)

        for _, row in df.loc[df['InterestHigh'] == 1].iterrows():
            df.loc[df.index > row.name, 'PreviousInterest'] = row['Interest']

        df['InterestScale'] = df['Interest'] / df['PreviousInterest']
        df['InterpAccuracy'] = self._calculate_for_accuracy(df)
        df['InterpPeaks'] = self._calculate_for_peaks(df)

        df['Result'] = np.nan
        df.loc[df['InterestScale'] < self._hybrid_separator, 'Result'] = df['InterpAccuracy']
        df.loc[df['InterestScale'] >= self._hybrid_separator, 'Result'] = df['InterpPeaks']

        df = df.loc[(date_start <= df['Date']) & (df['Date'] <= date_end)]
        df.reset_index(drop=True, inplace=True)

        ax[0].set_title(self.description)
        sns.lineplot(data=df, x='Date', y='Result', ax=ax[0])
        add_common_markers(df, ax[0])

        sns.lineplot(data=df, x='Date', y='InterpAccuracy', ax=ax[1])
        sns.lineplot(data=df, x='Date', y='InterpPeaks', ax=ax[1])
        add_common_markers(df, ax[1], price_line=False)

        return df['Result']
