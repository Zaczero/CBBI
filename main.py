import asyncio
import math
import time
import traceback
from pathlib import Path

import fire
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
from pyfiglet import figlet_format
from sty import bg, ef, fg, rs
from tqdm import tqdm

from fetch_bitcoin_data import fetch_bitcoin_data
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


def get_metrics():
    """
    Returns a list of available metrics to be calculated.
    """
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
    ]


def _json_number(value: float | None, *, precision: int = 4):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 'null'

    s = f'{value:.{precision}f}'.rstrip('0')
    if s.endswith('.'):
        s += '0'
    return s


def _write_columns_orient_json(df: pl.DataFrame, path: Path, *, precision: int = 4):
    ts = df.get_column('Date').dt.epoch(time_unit='s').to_list()
    cols = [c for c in df.columns if c != 'Date']

    with path.open('w', encoding='utf-8') as f:
        f.write('{\n')
        for col_i, col in enumerate(cols):
            f.write(f'  "{col}":{{\n')
            values = df.get_column(col).to_list()
            for row_i, (t, v) in enumerate(zip(ts, values, strict=True)):
                comma = ',' if row_i < len(ts) - 1 else ''
                f.write(f'    "{t}":{_json_number(v, precision=precision)}{comma}\n')
            col_comma = ',' if col_i < len(cols) - 1 else ''
            f.write(f'  }}{col_comma}\n')
        f.write('}')


def _add_common_markers(
    ax, *, halvings: np.ndarray, highs: np.ndarray, lows: np.ndarray
):
    for dt in halvings:
        ax.axvline(x=dt, color='navy', linestyle=':', linewidth=0.5)
    for dt in highs:
        ax.axvline(x=dt, color='green', linestyle=':', linewidth=0.5)
    for dt in lows:
        ax.axvline(x=dt, color='red', linestyle=':', linewidth=0.5)


def _shade_metric_bounds(ax):
    ax.axhline(y=1, color='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=0.5)

    y_min, y_max = ax.get_ylim()
    ax.fill_betweenx(
        y=[1, y_max],
        x1=0,
        x2=1,
        transform=ax.get_yaxis_transform(),
        color='black',
        alpha=0.1,
        edgecolor='none',
        zorder=0,
    )
    ax.fill_betweenx(
        y=[y_min, 0],
        x1=0,
        x2=1,
        transform=ax.get_yaxis_transform(),
        color='black',
        alpha=0.1,
        edgecolor='none',
        zorder=0,
    )


async def run(json_file: str, charts_file: str, output_dir: str | None):
    output_dir_path = Path.cwd() if output_dir is None else Path(output_dir)

    json_file_path = output_dir_path / Path(json_file)
    charts_file_path = output_dir_path / Path(charts_file)

    output_dir_path.mkdir(mode=0o755, parents=True, exist_ok=True)

    df_bitcoin = fetch_bitcoin_data()

    current_price = df_bitcoin.get_column('Price')[-1]
    print(
        'Current Bitcoin price: '
        + ef.b
        + fg.li_green
        + bg.da_green
        + f' $ {round(current_price):,} '
        + rs.all
    )

    metrics = get_metrics()
    metrics_cols = []
    metrics_descriptions = []

    sns.set_theme(
        font_scale=0.225,
        rc={
            'figure.titlesize': 12,
            'axes.titlesize': 7.5,
            'axes.labelsize': 6,
            'xtick.labelsize': 4,
            'ytick.labelsize': 4,
            'lines.linewidth': 0.5,
            'grid.linewidth': 0.3,
            'savefig.dpi': 1000,
            'figure.dpi': 300,
        },
    )

    x = df_bitcoin.get_column('Date').to_numpy()
    price_log = df_bitcoin.get_column('PriceLog').to_numpy()
    price = (price_log - price_log.min()) / (price_log.max() - price_log.min())

    halvings = x[df_bitcoin.get_column('Halving').to_numpy()]
    highs = x[df_bitcoin.get_column('PriceHigh').to_numpy()]
    lows = x[df_bitcoin.get_column('PriceLow').to_numpy()]

    axes_per_metric = 2
    fig, axes = plt.subplots(
        len(metrics), axes_per_metric, figsize=(4 * axes_per_metric, 3 * len(metrics))
    )
    axes = axes.reshape(-1, axes_per_metric)

    plt.tight_layout(pad=10)
    plt.subplots_adjust(top=0.98)
    plt.suptitle(
        'CBBI metric data input → output', fontsize=11.25, weight='bold', y=0.99508
    )

    for metric, ax_row in zip(metrics, axes, strict=True):
        ax_out = ax_row[1]
        ax_in = ax_row[0]

        sns.lineplot(x=x, y=price, alpha=0.4, color='orange', ax=ax_out)

        values = await metric.calculate(df_bitcoin, [ax_out, ax_in])

        _add_common_markers(ax_out, halvings=halvings, highs=highs, lows=lows)
        _add_common_markers(ax_in, halvings=halvings, highs=highs, lows=lows)

        _shade_metric_bounds(ax_out)

        ax_in.annotate(
            '',
            xy=(1.0967, 0.75),
            xycoords='axes fraction',
            xytext=(1.0367, 0.75),
            textcoords='axes fraction',
            arrowprops={
                'arrowstyle': '->',
                'color': 'darkgray',
                'lw': 1.5,
                'shrinkA': 0,
                'shrinkB': 0,
                'mutation_scale': 10,
            },
            ha='center',
            va='center',
        )

        values = values.clip(0, 1).rename(metric.name)
        df_bitcoin = df_bitcoin.with_columns(values)
        metrics_cols.append(metric.name)
        metrics_descriptions.append(metric.description)

    df_result = df_bitcoin.select('Date', 'Price', *metrics_cols).with_columns(
        Confidence=pl.mean_horizontal([pl.col(c).fill_nan(None) for c in metrics_cols])
    )

    print('Generating charts…')
    plt.savefig(charts_file_path)
    plt.close(fig)

    _write_columns_orient_json(df_result, json_file_path, precision=4)

    last = df_result.tail(1).row(0, named=True)
    confidence_details = {
        desc: last[name]
        for name, desc in zip(metrics_cols, metrics_descriptions, strict=True)
    }

    print('\n' + ef.b + ':: Confidence we are at the peak ::' + rs.all)
    print(
        fg.cyan
        + ef.bold
        + figlet_format(format_percentage(last['Confidence'], ''), font='univers')
        + rs.all,
        end='',
    )

    for description, value in confidence_details.items():
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            print(
                fg.white + get_color(value) + f'{format_percentage(value)} ' + rs.all,
                end='',
            )
            print(f' - {description}')

    print()
    print(
        'Source code: ' + ef.u + fg.li_blue + 'https://github.com/Zaczero/CBBI' + rs.all
    )
    print('License: ' + ef.b + 'AGPL-3.0' + rs.all)
    print()


def run_and_retry(
    json_file: str = 'latest.json',
    charts_file: str = 'charts.svg',
    output_dir: str | None = 'output',
    max_attempts: int = 3,
    sleep_seconds_on_error: int = 60,
):
    """
    Calculates the current CBBI confidence value alongside all the required metrics.
    Everything gets pretty printed to the current standard output and a clean copy
    is saved to a JSON file specified by the path in the ``json_file`` argument.
    A charts image is generated on the path specified by the ``charts_file`` argument
    which summarizes all individual metrics' historical data in a visual way.
    The execution is attempted multiple times in case an error occurs.

    Args:
        json_file: File path where the output is saved in the JSON format.
        charts_file: File path where the charts are saved (format inferred from file extension).
        output_dir: Directory path where the output is stored.
            If set to ``None`` then use the current working directory.
            If the directory does not exist, it will be created.
        max_attempts: Maximum number of attempts before termination. An attempt is counted when an error occurs.
        sleep_seconds_on_error: Duration of the sleep in seconds before attempting again after an error occurs.
    """
    assert max_attempts > 0, 'Value of the max_attempts argument must be positive'
    assert sleep_seconds_on_error >= 0, (
        'Value of the sleep_seconds_on_error argument must be non-negative'
    )

    for _ in range(max_attempts):
        try:
            asyncio.run(run(json_file, charts_file, output_dir))
            exit(0)

        except Exception:
            print(fg.black + bg.yellow + ' An error has occurred! ' + rs.all)
            traceback.print_exc()

            print(f'\nRetrying in {sleep_seconds_on_error} seconds…', flush=True)
            for _ in tqdm(range(sleep_seconds_on_error)):
                time.sleep(1)

    print(f'Max attempts limit has been reached ({max_attempts}).')
    print('Better luck next time!')
    exit(-1)


if __name__ == '__main__':
    fire.Fire(run_and_retry)
