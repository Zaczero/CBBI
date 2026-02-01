import pandas as pd

from utils import HTTP


def cs_fetch(path: str, data_selector: str, col_name: str) -> pd.DataFrame:
    url = f'https://coinank.com/indicatorapi/{path}'
    print(f'🔍 Fetching from Coinank API: {url}')
    
    response = HTTP.get(url)
    print(f'📡 Response status: {response.status_code}')
    response.raise_for_status()
    
    response_json = response.json()
    print(f'📦 Response JSON structure (first 500 chars): {str(response_json)[:500]}')
    
    data = response_json['data']

    if 'timeList' not in data and 'line' in data:
        print(f'🔄 Using nested "line" data structure')
        data = data['line']

    data_x = data['timeList']
    data_y = data[data_selector]
    print(f'📊 Data arrays: timeList length={len(data_x)}, {data_selector} length={len(data_y)}')
    assert len(data_x) == len(data_y), f'{len(data_x)=} != {len(data_y)=}'

    df = pd.DataFrame(
        {
            'Date': data_x[: len(data_y)],
            col_name: data_y,
        }
    )

    df['Date'] = pd.to_datetime(df['Date'], unit='ms').dt.tz_localize(None)
    print(f'✅ Successfully fetched {len(df)} rows for {col_name}')

    return df
