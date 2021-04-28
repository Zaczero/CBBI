import time
import traceback
from typing import List

import cli_ui
import fire
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from filecache import filecache
from matplotlib import pyplot as plt
from pyfiglet import figlet_format
from termcolor import cprint

from globals import HTTP_TIMEOUT
from metrics import BaseMetric, GoldenRatioMetric, GoogleTrendsMetric, StockToFlowMetric, PiCycleMetric, TwoYearMovingAverageMetric, TrolololoMetric, RUPLMetric, PuellMetric
from utils import mark_highs_lows, fix_block_halving_data, mark_days_since, format_percentage, get_color, fix_current_day_data

cli_ui.CONFIG['color'] = 'always'


# @filecache(3600 * 2)  # 2 hours cache
def fetch_bitcoin_data() -> pd.DataFrame:
    """
    Fetches Bitcoin data into a DataFrame for the past ``past_days`` days.
    The current day is skipped as the data for it is often incomplete (e.g. TotalGenerationUSD).

    Args:
        past_days: Amount of past days to fetch the Bitcoin data of.

    Returns:
        DataFrame containing Bitcoin data.
    """
    cli_ui.info_2('Requesting historical Bitcoin data')

    response = requests.get('https://api.blockchair.com/bitcoin/blocks', {
        'a': 'date,count(),min(id),max(id),sum(generation),sum(generation_usd)',
        's': 'date(desc)',
    }, timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    response_json = response.json()

    df = pd.DataFrame(response_json['data'][::-1])
    df.rename(columns={
        'date': 'Date',
        'count()': 'TotalBlocks',
        'min(id)': 'MinBlockID',
        'max(id)': 'MaxBlockID',
        'sum(generation)': 'TotalGeneration',
        'sum(generation_usd)': 'TotalGenerationUSD'
    }, inplace=True)

    df['Date'] = pd.to_datetime(df['Date'])
    df['TotalGeneration'] /= 1e8
    df['BlockGeneration'] = df['TotalGeneration'] / df['TotalBlocks']
    df['BlockGenerationUSD'] = df['TotalGenerationUSD'] / df['TotalBlocks']
    df['Price'] = df['BlockGenerationUSD'] / df['BlockGeneration']
    df['PriceLog'] = np.log(df['Price'])
    df = df[df['Date'] >= '2011-06-27']
    df.reset_index(drop=True, inplace=True)

    df = fix_current_day_data(df)
    df = fix_block_halving_data(df)
    df = mark_highs_lows(df, 'Price', False, round(365 * 2), 90)
    df = mark_days_since(df, ['PriceHigh', 'PriceLow', 'Halving'])

    current_price = df['Price'].tail(1).values[0]
    cli_ui.info_1(f'Current Bitcoin price: ${round(current_price):,}')

    return df


def load_metrics() -> List[BaseMetric]:
    return [
        GoldenRatioMetric(),
        GoogleTrendsMetric(),
        StockToFlowMetric(),
        PiCycleMetric(),
        TwoYearMovingAverageMetric(),
        TrolololoMetric(),
        RUPLMetric(),
        PuellMetric(),
    ]


def get_confidence_score(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    return df[cols].mean(axis=1)


def run(file: str) -> None:
    """
    Calculates the current CBBI confidence value alongside all the required metrics.
    Everything gets pretty printed to the current stdout and a clean copy
    is written to a JSON file specified by the path in the ``file`` argument.

    Args:
        file: File path where the output is stored in the JSON format.

    Returns:
        None
    """

    df_bitcoin = fetch_bitcoin_data()
    metrics = load_metrics()
    metrics_cols = []
    metrics_descriptions = []

    for metric in metrics:
        df_bitcoin[metric.name] = metric.calculate(df_bitcoin)
        metrics_cols.append(metric.name)
        metrics_descriptions.append(metric.description)

    confidence_col = 'Confidence'

    df_result = pd.DataFrame(df_bitcoin[['Date', 'Price'] + metrics_cols])
    df_result.set_index('Date', inplace=True)
    df_result[confidence_col] = get_confidence_score(df_result, metrics_cols)

    # sns.set()
    # _, ax = plt.subplots()
    # sns.lineplot(x='Date', y=confidence_col, data=df_result, ax=ax)
    # plt.show()

    df_result.to_json(file,
                      double_precision=4,
                      date_unit='s',
                      indent=2)

    df_result_last = df_result.tail(1)
    confidence_details = {description: df_result_last[name][0] for name, description in zip(metrics_cols, metrics_descriptions)}

    print('\n')
    cli_ui.info_3('Confidence we are at the peak:')
    cprint(figlet_format(format_percentage(df_result_last[confidence_col][0], ''), font='univers'), 'cyan', attrs=['bold'], end='')

    for description, value in confidence_details.items():
        cprint(format_percentage(value) + ' ', color=get_color(value), attrs=['reverse'], end='')
        print(f' - {description}')

    print()
    cli_ui.info_3('Source code: https://github.com/Zaczero/CBBI', end='\n\n')


def run_and_retry(file: str, max_attempts: int = 10, sleep_seconds_on_error: float = 10) -> None:
    """
    Calculates the current CBBI confidence value alongside all the required metrics.
    Everything gets pretty printed to the current stdout and a clean copy
    is written to a JSON file specified by the path in the ``file`` argument.
    The execution will be attempted multiple times in case an error occurs.

    Args:
        file: File path where the output is stored in the JSON format.
        max_attempts: Maximum number of attempts before termination. An attempt is counted when an error occurs.
        sleep_seconds_on_error: Duration of the sleep in seconds before attempting again after an error occurs.

    Returns:
        None
    """
    assert max_attempts > 0, 'Value of the max_attempts argument must be at least 1'
    assert sleep_seconds_on_error >= 0, 'Value of the sleep_seconds_on_error argument must be positive'

    for _ in range(max_attempts):
        try:
            run(file)
            exit(0)

        except Exception:
            cli_ui.error('An error occurred!')
            traceback.print_exc()

            print()
            cli_ui.info_1(f'Retrying in {sleep_seconds_on_error} seconds...')
            time.sleep(sleep_seconds_on_error)

    cli_ui.info_1(f'Max attempts limit has been reached ({max_attempts}). Better luck next time!')
    exit(-1)


if __name__ == '__main__':
    fire.Fire(run_and_retry)
