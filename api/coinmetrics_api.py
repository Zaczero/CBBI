import pandas as pd

from utils import HTTP


def cm_fetch_asset_metrics(
    *,
    asset: str,
    metrics: list[str],
    frequency: str = '1d',
    start_time: str,
    page_size: int = 10_000,
    null_as_zero: bool = True,
):
    """
    Fetch Coin Metrics Community time series asset metrics.

    Notes:
    - Coin Metrics returns numbers as strings. This helper casts metric columns to numeric.
    """
    response = HTTP.get(
        'https://community-api.coinmetrics.io/v4/timeseries/asset-metrics',
        params={
            'assets': asset,
            'metrics': ','.join(metrics),
            'frequency': frequency,
            'start_time': start_time,
            'paging_from': 'start',
            'page_size': page_size,
            'sort': 'time',
            'null_as_zero': str(null_as_zero).lower(),
        },
    )
    response.raise_for_status()
    response_json = response.json()
    data = response_json['data']

    while response_json.get('next_page_url'):
        response = HTTP.get(response_json['next_page_url'])
        response.raise_for_status()
        response_json = response.json()
        data.extend(response_json['data'])

    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['time']).dt.tz_localize(None).dt.floor('d')
    df.drop(columns=['time'], inplace=True)

    for col in metrics:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col])

    return df
