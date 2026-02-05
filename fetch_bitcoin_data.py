import polars as pl

from api.coinmetrics_api import cm_fetch_asset_metrics
from utils import mark_days_since, mark_highs_lows


def fetch_coinmetrics_data():
    """
    Fetch historical Bitcoin blockchain data from Coin Metrics Community API.
    """
    df_cm = cm_fetch_asset_metrics(
        asset='btc',
        metrics=['BlkCnt', 'IssTotNtv', 'IssTotUSD', 'PriceUSD'],
        start_time='2009-01-03',
    )

    df = df_cm.select(
        Date=pl.col('Date'),
        TotalBlocks=pl.col('BlkCnt'),
        IssTotNtv=pl.col('IssTotNtv'),
        IssTotUSD=pl.col('IssTotUSD'),
        Price=pl.col('PriceUSD'),
    )

    avg_subsidy = (
        pl
        .when(pl.col('TotalBlocks') > 0)
        .then(pl.col('IssTotNtv') / pl.col('TotalBlocks'))
        .otherwise(None)
    )
    subsidy_floor = pl.lit(50.0) / (
        pl.lit(2.0) ** (pl.lit(50.0) / avg_subsidy).log(base=2).ceil()
    )

    df = df.with_columns(
        Halving=(
            subsidy_floor.is_not_null()
            & subsidy_floor.shift(1).is_not_null()
            & (subsidy_floor != subsidy_floor.shift(1))
        )
    )

    return df.select(
        'Date',
        'IssTotUSD',
        'Price',
        'Halving',
    )


def fetch_bitcoin_data():
    """
    Fetches historical Bitcoin data into a DataFrame.
    Very early data is discarded due to high volatility.
    """
    print('📈 Requesting historical Bitcoin data…')

    df = fetch_coinmetrics_data()

    df = df.with_columns(
        PuellMultiple=(
            pl.col('IssTotUSD')
            / pl.col('IssTotUSD').rolling_mean(window_size=365, min_samples=365)
        ),
        Price730DMA=pl.col('Price').rolling_mean(window_size=730, min_samples=1),
    )

    df = df.filter(pl.col('Date') >= pl.datetime(2011, 6, 27, time_zone='UTC'))
    df = df.with_columns(
        PriceLog=pl.col('Price').log(),
    )

    df = mark_highs_lows(df, 'Price', False, 365 * 2, 180)

    # move 2021' peak to the first price peak
    df = df.with_columns(
        PriceHigh=(
            pl
            .when(pl.col('Date') == pl.datetime(2021, 11, 8, time_zone='UTC'))
            .then(False)
            .when(pl.col('Date') == pl.datetime(2021, 4, 14, time_zone='UTC'))
            .then(True)
            .otherwise(pl.col('PriceHigh'))
        )
    )

    return mark_days_since(df, ['PriceLow', 'Halving'])
