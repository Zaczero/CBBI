import pandas as pd
import requests

from globals import *


def lib_fetch(
        url_selector: str,
        post_selector: str,
        chart_idx: int,
        col_name: str
) -> pd.DataFrame:
    request_data = {
        'output': 'chart.figure',
        'changedPropIds': [
            'url.pathname'
        ],
        'inputs': [
            {
                'id': 'url',
                'property': 'pathname',
                'value': f'/charts/{post_selector}/'
            }
        ]
    }

    response = requests.post(
        f'https://www.lookintobitcoin.com/django_plotly_dash/app/{url_selector}/_dash-update-component',
        json=request_data,
        timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    response_json = response.json()
    response_x = response_json['response']['props']['figure']['data'][chart_idx]['x']
    response_y = response_json['response']['props']['figure']['data'][chart_idx]['y']

    df = pd.DataFrame({
        'Date': response_x[:len(response_y)],
        col_name: response_y,
    })
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    return df
