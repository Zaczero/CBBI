import filecache

from globals import *
from utils import *


@filecache.filecache(2 * filecache.HOUR)
def fetch_bitcoin_data() -> pd.DataFrame:
    """
    Fetches historical Bitcoin data into a DataFrame.
    Very early data is discarded due to high volatility.

    Returns:
        DataFrame containing Bitcoin data.
    """
    print('📈 Requesting historical Bitcoin data…')

    response = HTTP.get('https://api.blockchair.com/bitcoin/blocks', params={
        'a': 'date,count(),min(id),max(id),sum(generation),sum(generation_usd)',
        's': 'date(desc)',
    })
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

    df = df.merge(fetch_price_data(), on='Date', how='left')
    df.loc[df['Price'].isna(), 'Price'] = df['BlockGenerationUSD'] / df['BlockGeneration']
    df['PriceLog'] = np.log(df['Price'])
    df['PriceLogInterp'] = np.interp(df['PriceLog'],
                                     (df['PriceLog'].min(), df['PriceLog'].max()),
                                     (0, 1))

    df = df.loc[df['Date'] >= '2011-06-27']
    df.reset_index(drop=True, inplace=True)

    df = fix_current_day_data(df)
    df = add_block_halving_data(df)
    df = mark_highs_lows(df, 'Price', False, round(365 * 2), 180)
    df = mark_days_since(df, ['PriceHigh', 'PriceLow', 'Halving'])

    return df


def fetch_price_data() -> pd.DataFrame:
    response = HTTP.get('https://api.coinmarketcap.com/data-api/v3/cryptocurrency/detail/chart', params={
        'id': 1,
        'range': 'ALL',
    })

    response.raise_for_status()
    response_json = response.json()
    response_x = [float(k) for k in response_json['data']['points'].keys()]
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
        df.loc[(current_block_halving_id - reward_halving_every) <= df[
            'MaxBlockID'], 'BlockGeneration'] = current_block_production

        block_halving_row = df[(df['MinBlockID'] <= current_block_halving_id) &
                               (df['MaxBlockID'] >= current_block_halving_id)].squeeze()

        if block_halving_row.shape[0] == 0:
            break

        current_block_halving_id += reward_halving_every
        current_block_production /= 2
        df.loc[block_halving_row.name, 'Halving'] = 1
        df.loc[df.index > block_halving_row.name, 'NextHalvingBlock'] = current_block_halving_id

    df['DaysToHalving'] = pd.TimedeltaIndex((df['NextHalvingBlock'] - df['MaxBlockID']) / (24 * 6), unit='D')
    df['NextHalvingDate'] = df['Date'] + df['DaysToHalving']
    return df
