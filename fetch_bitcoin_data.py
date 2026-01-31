import numpy as np
import pandas as pd
from filecache import filecache

from utils import HTTP, http_get_with_retry, mark_days_since, mark_highs_lows

# Known Bitcoin halving dates and block heights for accurate calculations
HALVING_DATA = [
    # (date, block_height, block_reward)
    ('2009-01-03', 0, 50.0),           # Genesis block
    ('2012-11-28', 210000, 25.0),      # 1st halving
    ('2016-07-09', 420000, 12.5),      # 2nd halving
    ('2020-05-11', 630000, 6.25),      # 3rd halving
    ('2024-04-20', 840000, 3.125),     # 4th halving
]


def fetch_blockchain_data() -> pd.DataFrame:
    """
    Fetches historical Bitcoin blockchain data from Blockchain.com API.
    Uses miners-revenue chart for USD mining revenue.
    Block heights and BTC generation are calculated from halving schedule.
    
    Returns:
        DataFrame with Date, TotalBlocks, MinBlockID, MaxBlockID, 
        TotalGeneration, TotalGenerationUSD columns.
    """
    # Fetch mining revenue from Blockchain.com (free, reliable)
    response = http_get_with_retry(
        'https://api.blockchain.info/charts/miners-revenue',
        params={
            'timespan': 'all',
            'format': 'json',
            'sampled': 'false',
        },
    )
    revenue_data = response.json()
    
    # Create DataFrame from mining revenue data
    df = pd.DataFrame(revenue_data['values'])
    df.columns = ['DateTimestamp', 'TotalGenerationUSD']
    df['Date'] = pd.to_datetime(df['DateTimestamp'], unit='s').dt.floor('d')
    
    # Calculate block heights based on known halving dates
    # Average ~144 blocks per day (one block every 10 minutes)
    genesis_date = pd.Timestamp('2009-01-03')
    
    # Create halving schedule DataFrame for interpolation
    halving_df = pd.DataFrame(HALVING_DATA, columns=['Date', 'BlockHeight', 'BlockReward'])
    halving_df['Date'] = pd.to_datetime(halving_df['Date'])
    
    # Calculate approximate block height for each day using linear interpolation
    # between known halving points
    def estimate_block_height(date):
        date = pd.Timestamp(date)
        if date < genesis_date:
            return 0
        
        # Find the halving period this date falls into
        for i in range(len(HALVING_DATA) - 1):
            start_date = pd.Timestamp(HALVING_DATA[i][0])
            end_date = pd.Timestamp(HALVING_DATA[i + 1][0])
            start_height = HALVING_DATA[i][1]
            end_height = HALVING_DATA[i + 1][1]
            
            if start_date <= date < end_date:
                # Linear interpolation within this halving period
                total_days = (end_date - start_date).days
                days_elapsed = (date - start_date).days
                height = start_height + (end_height - start_height) * days_elapsed / total_days
                return int(height)
        
        # After the last known halving, extrapolate at ~144 blocks/day
        last_date = pd.Timestamp(HALVING_DATA[-1][0])
        last_height = HALVING_DATA[-1][1]
        days_since = (date - last_date).days
        return int(last_height + days_since * 144)
    
    def get_block_reward(block_height):
        """Get block reward for a given block height."""
        halving_interval = 210000
        halvings = block_height // halving_interval
        return 50.0 / (2 ** halvings)
    
    # Calculate block data for each day
    df['MaxBlockID'] = df['Date'].apply(estimate_block_height)
    df['MinBlockID'] = df['MaxBlockID'].shift(1).fillna(0).astype(int)
    df['TotalBlocks'] = df['MaxBlockID'] - df['MinBlockID']
    df['TotalBlocks'] = df['TotalBlocks'].clip(lower=1)  # Ensure at least 1 block
    
    # Calculate BTC generation based on block reward
    # Store in satoshis (multiply by 1e8) to match original Blockchair format
    df['BlockReward'] = df['MaxBlockID'].apply(get_block_reward)
    df['TotalGeneration'] = df['TotalBlocks'] * df['BlockReward'] * 1e8  # Convert to satoshis
    
    # Select and order columns to match original format
    df = df[['Date', 'TotalBlocks', 'MinBlockID', 'MaxBlockID', 'TotalGeneration', 'TotalGenerationUSD']]
    df = df.sort_values('Date').reset_index(drop=True)
    
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

    # Use Blockchain.com API instead of Blockchair (which is blocked)
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
    df = add_block_halving_data(df)
    df = mark_highs_lows(df, 'Price', False, round(365 * 2), 180)

    # move 2021' peak to the first price peak
    df.loc[df['Date'] == '2021-11-09', 'PriceHigh'] = 0
    df.loc[df['Date'] == '2021-04-14', 'PriceHigh'] = 1

    df = mark_days_since(df, ['PriceHigh', 'PriceLow', 'Halving'])
    return df


def fetch_price_data() -> pd.DataFrame:
    response = http_get_with_retry(
        'https://api.coinmarketcap.com/data-api/v3/cryptocurrency/detail/chart',
        params={
            'id': 1,
            'range': 'ALL',
        },
    )

    response_json = response.json()
    response_x = [float(k) for k in response_json['data']['points']]
    response_y = [value['v'][0] for value in response_json['data']['points'].values()]

    df = pd.DataFrame(
        {
            'Date': response_x,
            'Price': response_y,
        }
    )
    df['Date'] = pd.to_datetime(df['Date'], unit='s').dt.tz_localize(None).dt.floor('d')
    df.sort_values(by='Date', inplace=True)
    df.drop_duplicates('Date', keep='last', inplace=True)

    return df


def fix_current_day_data(df: pd.DataFrame) -> pd.DataFrame:
    row = df.iloc[-1].copy()

    target_total_blocks = 24 * 6
    target_scale = target_total_blocks / row['TotalBlocks']

    for col_name in ['TotalBlocks', 'TotalGeneration', 'TotalGenerationUSD']:
        row[col_name] *= target_scale

    df.iloc[-1] = row
    return df


def add_block_halving_data(df: pd.DataFrame) -> pd.DataFrame:
    reward_halving_every = 210000
    current_block_halving_id = reward_halving_every
    current_block_production = 50
    df['Halving'] = 0
    df['NextHalvingBlock'] = current_block_halving_id

    while True:
        df.loc[
            (current_block_halving_id - reward_halving_every) <= df['MaxBlockID'],
            'BlockGeneration',
        ] = current_block_production

        block_halving_rows = df[
            (df['MinBlockID'] <= current_block_halving_id) & (df['MaxBlockID'] >= current_block_halving_id)
        ]

        if len(block_halving_rows) == 0:
            break

        # Take the first matching row if multiple match
        block_halving_row = block_halving_rows.iloc[0]
        row_index = block_halving_rows.index[0]

        current_block_halving_id += reward_halving_every
        current_block_production /= 2
        df.loc[row_index, 'Halving'] = 1
        df.loc[df.index > row_index, 'NextHalvingBlock'] = current_block_halving_id

    df['DaysToHalving'] = pd.to_timedelta((df['NextHalvingBlock'] - df['MaxBlockID']) / (24 * 6), unit='D')
    df['NextHalvingDate'] = df['Date'] + df['DaysToHalving']
    return df
