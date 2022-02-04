import time
import traceback

import fire
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pyfiglet import figlet_format
from sty import fg, bg, rs, ef
from tqdm import tqdm

from fetch_bitcoin_data import fetch_bitcoin_data
from metrics.base_metric import BaseMetric
from metrics.google_trends import GoogleTrendsMetric
from metrics.halving_to_peak import HalvingToPeakMetric
from metrics.mvrv_z_score import MVRVMetric
from metrics.pi_cycle import PiCycleMetric
from metrics.puell_multiple import PuellMetric
from metrics.reserve_risk import ReserveRiskMetric
from metrics.rhodl_ratio import RHODLMetric
from metrics.rupl import RUPLMetric
from metrics.trolololo import TrolololoMetric
from metrics.two_year_moving_average import TwoYearMovingAverageMetric
from metrics.woobull_topcap_cvdd import WoobullMetric
from utils import format_percentage, get_color


def get_metrics() -> list[BaseMetric]:
    return [
        PiCycleMetric(),
        RUPLMetric(),
        RHODLMetric(),
        PuellMetric(),
        TwoYearMovingAverageMetric(),
        TrolololoMetric(),
        MVRVMetric(),
        ReserveRiskMetric(),
        WoobullMetric(),
        HalvingToPeakMetric(),
        GoogleTrendsMetric(),
    ]


def calculate_confidence_score(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols].mean(axis=1)


def run(json_file: str, charts_file: str) -> None:
    """
    Calculates the current CBBI confidence value alongside all the required metrics.
    Everything gets pretty printed to the current standard output and a clean copy
    is saved to a JSON file specified by the path in the ``json_file`` argument.
    A charts image is generated on the path specified by the ``charts_file`` argument
    which summarizes all individual metrics' historical data in a visual way.

    Args:
        json_file: File path where the output is saved in the JSON format.
        charts_file: File path where the charts image is saved (formats supported by pyplot.savefig).

    Returns:
        None
    """

    df_bitcoin = fetch_bitcoin_data()
    df_bitcoin_org = df_bitcoin.copy()

    current_price = df_bitcoin['Price'].tail(1).values[0]
    print(f'Current Bitcoin price: ' + ef.b + fg.li_green + bg.da_green + f' $ {round(current_price):,} ' + rs.all)

    metrics = get_metrics()
    metrics_cols = []
    metrics_descriptions = []

    sns.set(font_scale=0.15, rc={
        # 'font.size': 6,
        'figure.titlesize': 8,
        'axes.titlesize': 5,
        'axes.labelsize': 4,
        'xtick.labelsize': 4,
        'ytick.labelsize': 4,
        'lines.linewidth': 0.5,
        'grid.linewidth': 0.3,

        'savefig.dpi': 1000,
        'figure.dpi': 300,
    })

    axes_per_metric = 2
    fig, axes = plt.subplots(len(metrics), axes_per_metric, figsize=(4 * axes_per_metric, 3 * len(metrics)))
    axes = axes.reshape(-1, axes_per_metric)
    plt.tight_layout(pad=14)

    for metric, ax in zip(metrics, axes):
        df_bitcoin[metric.name] = metric.calculate(df_bitcoin_org.copy(), ax).clip(0, 1)
        metrics_cols.append(metric.name)
        metrics_descriptions.append(metric.description)

    print('Generating charts…')
    plt.savefig(charts_file)

    confidence_col = 'Confidence'

    df_result = pd.DataFrame(df_bitcoin[['Date', 'Price'] + metrics_cols])
    df_result.set_index('Date', inplace=True)
    df_result[confidence_col] = calculate_confidence_score(df_result, metrics_cols)
    df_result \
        .to_json(json_file,
                 double_precision=4,
                 date_unit='s',
                 indent=2)

    df_result_last = df_result.tail(1)
    confidence_details = {description: df_result_last[name][0]
                          for name, description in
                          zip(metrics_cols, metrics_descriptions)}

    print('\n' + ef.b + ':: Confidence we are at the peak ::' + rs.all)
    print(
        fg.cyan + ef.bold +
        figlet_format(
            format_percentage(df_result_last[confidence_col][0], ''),
            font='univers') + rs.all,
        end='')

    for description, value in confidence_details.items():
        if not np.isnan(value):
            print(fg.white + get_color(value) + f'{format_percentage(value)} ' + rs.all, end='')
            print(f' - {description}')

    print('\nSource code: ' + ef.u + fg.li_blue + 'https://github.com/Zaczero/CBBI' + rs.all + '\n')


def run_and_retry(json_file: str = 'latest.json',
                  charts_file: str = 'charts.svg',
                  max_attempts: int = 10,
                  sleep_seconds_on_error: int = 10) -> None:
    """
    Calculates the current CBBI confidence value alongside all the required metrics.
    Everything gets pretty printed to the current standard output and a clean copy
    is saved to a JSON file specified by the path in the ``json_file`` argument.
    A charts image is generated on the path specified by the ``charts_file`` argument
    which summarizes all individual metrics' historical data in a visual way.
    The execution is attempted multiple times in case an error occurs.

    Args:
        json_file: File path where the output is saved in the JSON format.
        charts_file: File path where the charts image is saved (formats supported by pyplot.savefig).
        max_attempts: Maximum number of attempts before termination. An attempt is counted when an error occurs.
        sleep_seconds_on_error: Duration of the sleep in seconds before attempting again after an error occurs.

    Returns:
        None
    """
    assert max_attempts > 0, 'Value of the max_attempts argument must be positive'
    assert sleep_seconds_on_error >= 0, 'Value of the sleep_seconds_on_error argument must be non-negative'

    for _ in range(max_attempts):
        try:
            run(json_file, charts_file)
            exit(0)

        except Exception:
            print(fg.black + bg.yellow + ' An error has occurred! ' + rs.all)
            traceback.print_exc()

            print(f'\nRetrying in {sleep_seconds_on_error} seconds…', flush=True)
            for _ in tqdm(range(sleep_seconds_on_error)):
                time.sleep(1)

    print(f'Max attempts limit has been reached ({max_attempts}).')
    print(f'Better luck next time!')
    exit(-1)


if __name__ == '__main__':
    fire.Fire(run_and_retry)
