import polars as pl

from utils import HTTP


def cbbi_fetch(key: str):
    response = HTTP.get('https://colintalkscrypto.com/cbbi/data/latest.json')
    response.raise_for_status()
    response_data = response.json()[key]

    return (
        pl
        .DataFrame({
            'Date': pl.Series([int(k) for k in response_data], dtype=pl.Int64),
            'Value': pl.Series(list(response_data.values()), dtype=pl.Float64),
        })
        .with_columns(
            Date=(
                pl
                .from_epoch(pl.col('Date'), time_unit='s')
                .dt.cast_time_unit('us')
                .dt.replace_time_zone('UTC')
            ),
        )
        .select('Date', 'Value')
    )
