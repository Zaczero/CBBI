import pandas as pd

from utils import HTTP


def cs_fetch(path: str, data_selector: str, col_name: str) -> pd.DataFrame:
    response = HTTP.get(f'https://coinsoto.com/indicatorapi/{path}')
    response.raise_for_status()
    data = response.json()['data']

    if 'timeList' not in data and 'line' in data:
        data = data['line']

    data_x = data['timeList']
    data_y = data[data_selector]
    assert len(data_x) == len(data_y), f'{len(data_x)=} != {len(data_y)=}'

    df = pd.DataFrame(
        {
            'Date': data_x[: len(data_y)],
            col_name: data_y,
        }
    )

    df['Date'] = pd.to_datetime(df['Date'], unit='ms').dt.tz_localize(None)

    return df
