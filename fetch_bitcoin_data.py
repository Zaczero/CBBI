from itertools import count, pairwise

import numpy as np
import pandas as pd
from filecache import filecache

from utils import HTTP, mark_days_since, mark_highs_lows

HALVING_INTERVAL = 210_000
GENESIS_BLOCK_REWARD = 50.0
BLOCKS_PER_DAY = 144


def fetch_block_halving():
    """
    Fetch Bitcoin halving data by probing raw blocks at halving heights.
    """
    halving_data: list[tuple[pd.Timestamp, int, float]] = []

    for halving_index in count():
        block_height = halving_index * HALVING_INTERVAL
        block_reward = GENESIS_BLOCK_REWARD / (2**halving_index)

        response = HTTP.get(f'https://blockchain.info/rawblock/{block_height}')
        if response.status_code == 404:
            break

        response.raise_for_status()
        block_time = response.json()['time']
        block_date = pd.to_datetime(block_time, unit='s').tz_localize(None).floor('d')
        halving_data.append((block_date, block_height, block_reward))

    return halving_data


def fetch_blockchain_data() -> pd.DataFrame:
    """
    Fetches historical Bitcoin blockchain data from Blockchain.com API.
    Uses miners-revenue chart for USD mining revenue.
    Block heights and BTC generation are calculated from halving schedule.

    Returns:
        DataFrame with Date, TotalBlocks, MinBlockID, MaxBlockID,
        TotalGeneration, TotalGenerationUSD columns.
    """
    halving_data = fetch_block_halving()

    # Fetch mining revenue from Blockchain.com
    response = HTTP.get(
        'https://api.blockchain.info/charts/miners-revenue',
        params={
            'timespan': 'all',
            'format': 'json',
            'sampled': 'false',
        },
    )

    # Create DataFrame from mining revenue data
    response.raise_for_status()
    df = pd.DataFrame(response.json()['values'])
    df.columns = ['DateTimestamp', 'TotalGenerationUSD']
    df['Date'] = pd.to_datetime(df['DateTimestamp'], unit='s').dt.floor('d')

    # Calculate approximate block height for each day using linear interpolation
    # between known halving points
    def estimate_block_height(date: pd.Timestamp):
        # Find the halving period this date falls into
        for (start_date, start_height, _), (end_date, end_height, _) in pairwise(halving_data):
            if start_date <= date < end_date:
                # Linear interpolation within this halving period
                total_days = (end_date - start_date).days
                days_elapsed = (date - start_date).days
                height = start_height + (end_height - start_height) * days_elapsed / total_days
                return int(height)

        # After the last known halving, extrapolate
        (last_date, last_height, _) = halving_data[-1]
        days_since = (date - last_date).days
        return int(last_height + days_since * BLOCKS_PER_DAY)

    def get_block_reward(block_height):
        """Get block reward for a given block height."""
        halvings = block_height // HALVING_INTERVAL
        return GENESIS_BLOCK_REWARD / (2**halvings)

    # Calculate block data for each day
    df['MaxBlockID'] = df['Date'].apply(estimate_block_height)
    df['MinBlockID'] = df['MaxBlockID'].shift(1, fill_value=0)
    df['TotalBlocks'] = df['MaxBlockID'] - df['MinBlockID']

    # Calculate BTC generation based on block reward
    df['BlockReward'] = df['MaxBlockID'].apply(get_block_reward)
    df['TotalGeneration'] = df['TotalBlocks'] * df['BlockReward'] * 1e8  # Convert to satoshis

    # Select and order columns to match original format
    df = df[['Date', 'TotalBlocks', 'MinBlockID', 'MaxBlockID', 'TotalGeneration', 'TotalGenerationUSD']]
    df = df.sort_values('Date').reset_index(drop=True)

    # Add halving markers
    df['Halving'] = 0
    for _, block_height, _ in halving_data[1:]:
        df.loc[(df['MinBlockID'] < block_height) & (df['MaxBlockID'] >= block_height), 'Halving'] = 1

    return df


@filecache(7200)  # 2 hours
def fetch_bitcoin_data() -> pd.DataFrame:
    """
    Fetches historical Bitcoin data into a DataFrame.
    Very early data is discarded due to high volatility.

    Returns:
        DataFrame containing Bitcoin data.
    """
    print('📈 Requesting historical Bitcoin data…')

    df = fetch_blockchain_data()

    df['Date'] = pd.to_datetime(df['Date'])
    df['TotalGeneration'] /= 1e8
    df['BlockGeneration'] = df['TotalGeneration'] / df['TotalBlocks']
    df['BlockGenerationUSD'] = df['TotalGenerationUSD'] / df['TotalBlocks']

    df = df.merge(fetch_price_data(), on='Date', how='left')
    df.loc[df['Price'].isna(), 'Price'] = df['BlockGenerationUSD'] / df['BlockGeneration']
    df['PriceLog'] = np.log(df['Price'])
    df['PriceLogInterp'] = np.interp(
        x=df['PriceLog'],
        xp=(df['PriceLog'].min(), df['PriceLog'].max()),
        fp=(0, 1),
    )

    df = df.loc[df['Date'] >= '2011-06-27']
    df.reset_index(drop=True, inplace=True)

    df = fix_current_day_data(df)
    df = mark_highs_lows(df, 'Price', False, round(365 * 2), 180)

    # move 2021' peak to the first price peak
    df.loc[df['Date'] == '2021-11-09', 'PriceHigh'] = 0
    df.loc[df['Date'] == '2021-04-14', 'PriceHigh'] = 1

    df = mark_days_since(df, ['PriceHigh', 'PriceLow', 'Halving'])
    return df


def fetch_price_data() -> pd.DataFrame:
    response = HTTP.get(
        'https://api.coinmarketcap.com/data-api/v3/cryptocurrency/detail/chart',
        params={
            'id': 1,
            'range': 'ALL',
        },
    )

    response.raise_for_status()
    response_json = response.json()
    response_x = [float(k) for k in response_json['data']['points']]
    response_y = [value['v'][0] for value in response_json['data']['points'].values()]

    df = pd.DataFrame({
        'Date': response_x,
        'Price': response_y,
    })
    df['Date'] = pd.to_datetime(df['Date'], unit='s').dt.tz_localize(None).dt.floor('d')
    df.sort_values(by='Date', inplace=True)
    df.drop_duplicates('Date', keep='last', inplace=True)

    return df


def fix_current_day_data(df: pd.DataFrame) -> pd.DataFrame:
    row = df.iloc[-1].copy()

    target_scale = BLOCKS_PER_DAY / row['TotalBlocks']

    for col_name in ['TotalBlocks', 'TotalGeneration', 'TotalGenerationUSD']:
        row[col_name] *= target_scale

    df.iloc[-1] = row
    return df
