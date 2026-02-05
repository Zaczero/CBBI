import polars as pl

from utils import HTTP


def cs_fetch(path: str, data_selector: str, col_name: str):
    response = HTTP.get(f'https://api.coinank.com/indicatorapi/{path}')
    response.raise_for_status()
    data = response.json()['data']

    if 'timeList' not in data and 'line' in data:
        data = data['line']

    data_x = data['timeList']
    data_y = data[data_selector]

    return (
        pl
        .DataFrame({
            'Date': data_x[: len(data_y)],
            col_name: data_y,
        })
        .with_columns(
            Date=(
                pl
                .from_epoch(pl.col('Date'), time_unit='ms')
                .dt.cast_time_unit('us')
                .dt.replace_time_zone('UTC')
            )
        )
        .select('Date', col_name)
    )
