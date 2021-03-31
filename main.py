from typing import Tuple

import numpy as np
import pandas as pd
import requests
from pytrends.request import TrendReq
from pyfiglet import figlet_format
from termcolor import cprint
import cli_ui


def mark_highs_lows(df: pd.DataFrame, col: str, begin_with_high: bool, window_size: float) -> pd.DataFrame:
    df['High'] = 0
    df['Low'] = 0

    searching_high = begin_with_high
    current_index = df.index[0]

    while True:
        window = df.loc[current_index:current_index + window_size, col]
        window_index = window.idxmax() if searching_high else window.idxmin()

        if window.shape[0] == 1:
            break

        if window_index == current_index:
            df.loc[window_index, 'High' if searching_high else 'Low'] = 1
            searching_high = not searching_high

        current_index = window_index

    return df


def fetch_bitcoin_data(past_days: int) -> pd.DataFrame:
    cli_ui.info_2(f'Fetching Bitcoin data for the past {past_days} days')

    response = requests.get('https://api.blockchair.com/bitcoin/blocks', {
        'a': 'date,count(),min(id),max(id),sum(generation),sum(generation_usd)',
        's': 'date(desc)',
        'limit': past_days + 1,
    }, timeout=30)
    response.raise_for_status()
    response_json = response.json()

    df = pd.DataFrame(response_json['data'][::-1], )
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

    current_price = df['Price'].tail(1).values[0]
    cli_ui.info_1(f'Current Bitcoin price: ${round(current_price):,}')

    return df.head(past_days)


def get_golden_ratio_index(df: pd.DataFrame) -> float:
    current_bottom_date = pd.to_datetime('2018-12-03')
    current_halving_date = pd.to_datetime('2020-05-11')
    current_peak_date = current_bottom_date + (current_halving_date - current_bottom_date) / .51
    current_date = df['Date'].tail(1).values[0]

    return (current_date - current_bottom_date) / (current_peak_date - current_bottom_date)


def get_pi_cycle_index(df: pd.DataFrame) -> float:
    max_divergence_price_high = np.log(12000)
    max_divergence_price_low = np.log(3700)
    max_divergence = max_divergence_price_high - max_divergence_price_low

    df['111DMA'] = df['Price'].rolling(111).mean()
    df['350DMAx2'] = df['Price'].rolling(350).mean() * 2

    df['111DMALog'] = np.log(df['111DMA'])
    df['350DMAx2Log'] = np.log(df['350DMAx2'])
    df['PiCycleDifference'] = np.abs(df['111DMALog'] - df['350DMAx2Log'])

    return (1 - df['PiCycleDifference'] / max_divergence).tail(1).values[0]


def get_2yma_index(df: pd.DataFrame) -> float:
    overshoot = .2

    df['2YMA'] = df['Price'].rolling(365 * 2).mean()
    df['2YMAx5'] = df['2YMA'] * 5

    df['2YMALog'] = np.log(df['2YMA'])
    df['2YMAx5Log'] = np.log(df['2YMAx5'])
    df['2YMALogDifference'] = df['2YMAx5Log'] - df['2YMALog']
    df['2YMALogOvershootDifference'] = df['2YMALogDifference'] * overshoot
    df['2YMALogOvershoot'] = df['2YMAx5Log'] + df['2YMALogOvershootDifference']
    df['2YMALogUndershoot'] = df['2YMALog'] - df['2YMALogOvershootDifference']

    return ((df['PriceLog'] - df['2YMALogUndershoot']) / (df['2YMALogOvershoot'] - df['2YMALogUndershoot'])).tail(1).values[0]


def get_puell_index(df: pd.DataFrame) -> float:
    projected_max = np.log(4.75)
    projected_min = np.log(0.3)

    df['PuellMA365'] = df['TotalGenerationUSD'].rolling(365).mean()
    df['Puell'] = df['TotalGenerationUSD'] / df['PuellMA365']
    df['PuellLog'] = np.log(df['Puell'])

    return ((df['PuellLog'] - projected_min) / (projected_max - projected_min)).tail(1).values[0]


def get_sf_index(df: pd.DataFrame) -> float:
    previous_hinge_date = pd.to_datetime('2017-10-15')
    previous_peak_date = pd.to_datetime('2017-12-15')
    current_hinge_date = pd.to_datetime('2021-08-15')
    current_peak_date = current_hinge_date + (previous_peak_date - previous_hinge_date)
    current_date = df['Date'].tail(1).values[0]

    return (current_date - previous_peak_date) / (current_peak_date - previous_peak_date)


def get_rupl_index() -> float:
    projected_max = .75
    projected_min = -.2

    cli_ui.info_2('Fetching RUPL data')

    response = requests.get('https://www.lookintobitcoin.com/django_plotly_dash/app/unrealised_profit_loss/_dash-layout', timeout=30)
    response.raise_for_status()
    response_json = response.json()

    current_value = response_json['props']['children'][0]['props']['figure']['data'][0]['y'][-1]

    return (current_value - projected_min) / (projected_max - projected_min)


def get_google_trends_index(manual: bool) -> float:
    target_ratio = 1 / .125

    if manual:
        previous_peak = int(cli_ui.ask_string('Enter the previous Google Trends peak'))
        current_peak = int(cli_ui.ask_string('Enter the current Google Trends peak'))
    else:
        cli_ui.info_2('Fetching Google Trends data')

        pytrends = TrendReq()
        pytrends.build_payload(kw_list=['Bitcoin'])
        df_interest = pytrends.interest_over_time()
        df_interest.reset_index(inplace=True)
        df_interest.rename(columns={
            'date': 'Date',
            'Bitcoin': 'Interest'
        }, inplace=True)
        df_interest = mark_highs_lows(df_interest, 'Interest', True, round(365 * 2 / 7))

        previous_peak = df_interest.loc[df_interest['High'] == 1].head(1)['Interest'].values[0]
        current_peak = df_interest.tail(1)['Interest'].values[0]

        cli_ui.info_1(f'Previous Google Trends peak: {previous_peak}%')
        cli_ui.info_1(f'Current Google Trends peak: {current_peak}%')

    current_ratio = current_peak / previous_peak
    return current_ratio / target_ratio


def format_percentage(val: float) -> str:
    return f'{round(val * 100): >3d} %'


def get_color(val: float) -> str:
    bin_map = [
        'red',
        'yellow',
        'green',
        'cyan',
        'magenta',
    ]
    bins = [
        .3,
        .65,
        .85,
        .97,
    ]

    bin_index = np.digitize([round(val, 2)], bins)[0]
    return bin_map[bin_index]


if __name__ == '__main__':
    df_data = fetch_bitcoin_data(365 * 2)

    golden_ratio_index = get_golden_ratio_index(df_data)
    pi_cycle_index = get_pi_cycle_index(df_data)
    _2yma_index = get_2yma_index(df_data)
    puell_index = get_puell_index(df_data)

    sf_index = get_sf_index(df_data)
    rupl_index = get_rupl_index()
    google_trends_index = get_google_trends_index(False)

    confidence = np.mean([
        golden_ratio_index,
        google_trends_index,
        sf_index,
        pi_cycle_index,
        _2yma_index,
        rupl_index,
        puell_index
    ])

    details = {
        'The Golden 51%-49% Ratio': golden_ratio_index,
        '"Bitcoin" search term (Google Trends)': google_trends_index,
        'Stock-to-Flow Chart': sf_index,
        'Pi Cycle Top Indicator': pi_cycle_index,
        'Bitcoin Investor Tool: 2-Year Moving Average': _2yma_index,
        'NUPL a.k.a. RUPL': rupl_index,
        'Puell Multiple': puell_index,
    }

    print('\n')
    cli_ui.info_3('Confidence we are at the peak:')
    cprint(figlet_format(format_percentage(confidence), font='univers'), 'cyan', attrs=['bold'], end='')

    for k, v in details.items():
        cprint(format_percentage(v) + ' ', color=get_color(v), attrs=['reverse'], end='')
        print(f' - {k}')

    print('\n')
    cli_ui.info_3('Source code: https://github.com/Zaczero/CBBI')
