import pandas as pd
import requests

from globals import *


def lib_fetch(
        url_selector: str,
        post_selector: str,
        chart_idx: str,
        col_name: str
) -> pd.DataFrame:
    request_data = {
        'changedPropIds': [
            'url.pathname'
        ],
        'inputs': [
            {
                'id': 'url',
                'property': 'pathname',
                'value': f'/charts/{post_selector}/'
            }
        ],
        'output': 'chart.figure',
        'outputs': {
            'id': 'chart',
            'property': 'figure',
        },
    }

    response = requests.post(
        f'https://www.lookintobitcoin.com/django_plotly_dash/app/{url_selector}/_dash-update-component',
        json=request_data,
        timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    response_json = response.json()

    data = next(v
                for v in response_json['response']['chart']['figure']['data']
                if 'name' in v and v['name'] == chart_idx)

    data_x, data_y = data['x'], data['y']

    df = pd.DataFrame({
        'Date': data_x[:len(data_y)],
        col_name: data_y,
    })
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    return df
