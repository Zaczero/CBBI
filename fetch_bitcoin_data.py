import numpy as np
import pandas as pd
from filecache import filecache

from api.coinmetrics_api import cm_fetch_asset_metrics
from utils import mark_days_since, mark_highs_lows


def fetch_coinmetrics_data():
    """
    Fetch historical Bitcoin blockchain data from Coin Metrics Community API.

    Returns:
        DataFrame with Date, TotalBlocks, MinBlockID, MaxBlockID,
        TotalGeneration, TotalGenerationUSD columns.
    """
    df_cm = cm_fetch_asset_metrics(
        asset='btc',
        metrics=['BlkCnt', 'FeeTotNtv', 'IssTotNtv', 'IssTotUSD', 'PriceUSD'],
        start_time='2009-01-03',
        null_as_zero=True,
    )

    df = pd.DataFrame({
        'Date': df_cm['Date'],
        'TotalBlocks': df_cm['BlkCnt'],
        'FeeTotNtv': df_cm['FeeTotNtv'],
        'IssTotNtv': df_cm['IssTotNtv'],
        'IssTotUSD': df_cm['IssTotUSD'],
        'Price': df_cm['PriceUSD'],
    })

    df['MaxBlockID'] = df['TotalBlocks'].cumsum() - 1
    df['MinBlockID'] = df['MaxBlockID'] - df['TotalBlocks'] + 1

    df['TotalGeneration'] = df['FeeTotNtv'] + df['IssTotNtv']
    df['TotalGenerationUSD'] = df['IssTotUSD'] + df['FeeTotNtv'] * df['Price']

    avg_subsidy = df['IssTotNtv'] / df['TotalBlocks'].where(df['TotalBlocks'] > 0)
    subsidy_floor = 50 / (2 ** np.ceil(np.log2(50 / avg_subsidy)))
    df['Halving'] = (
        subsidy_floor.notna() & subsidy_floor.shift(1).notna() & (subsidy_floor != subsidy_floor.shift(1))
    ).astype(int)

    return df[
        ['Date', 'TotalBlocks', 'MinBlockID', 'MaxBlockID', 'TotalGeneration', 'TotalGenerationUSD', 'Price', 'Halving']
    ]


@filecache(7200)  # 2 hours
def fetch_bitcoin_data() -> pd.DataFrame:
    """
    Fetches historical Bitcoin data into a DataFrame.
    Very early data is discarded due to high volatility.

    Returns:
        DataFrame containing Bitcoin data.
    """
    print('📈 Requesting historical Bitcoin data…')

    df = fetch_coinmetrics_data()

    df['Date'] = pd.to_datetime(df['Date'])

    df = df.loc[df['Date'] >= '2011-06-27']
    df.reset_index(drop=True, inplace=True)

    df['BlockGeneration'] = df['TotalGeneration'] / df['TotalBlocks']
    df['BlockGenerationUSD'] = df['TotalGenerationUSD'] / df['TotalBlocks']
    df['PriceLog'] = np.log(df['Price'])
    df['PriceLogInterp'] = np.interp(
        x=df['PriceLog'],
        xp=(df['PriceLog'].min(), df['PriceLog'].max()),
        fp=(0, 1),
    )

    df = mark_highs_lows(df, 'Price', False, round(365 * 2), 180)

    # move 2021' peak to the first price peak
    df.loc[df['Date'] == '2021-11-08', 'PriceHigh'] = 0
    df.loc[df['Date'] == '2021-04-14', 'PriceHigh'] = 1

    df = mark_days_since(df, ['PriceHigh', 'PriceLow', 'Halving'])
    return df
