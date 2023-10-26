import pandas as pd

from globals import *
from utils import HTTP


def lib_fetch(
        url_selector: str,
        post_selector: str,
        chart_idx: str,
        col_name: str,
        configurable: bool = False,
) -> pd.DataFrame:
    if not configurable:
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
    else:
        request_data = {
            'changedPropIds': [
                'url.pathname'
            ],
            'inputs': [
                {
                    'id': 'url',
                    'property': 'pathname',
                    'value': f'/charts/{post_selector}/'
                },
                {
                    "id": "resolution",
                    "property": "value",
                    "value": "24h"
                },
                {
                    "id": "scale",
                    "property": "value",
                    "value": "log"
                }
            ],
            'output': '..chart.figure...resolution.disabled...resolution.value...scale.disabled...scale.value..',
            'outputs': [
                {
                    "id": "chart",
                    "property": "figure"
                },
                {
                    "id": "resolution",
                    "property": "disabled"
                },
                {
                    "id": "resolution",
                    "property": "value"
                },
                {
                    "id": "scale",
                    "property": "disabled"
                },
                {
                    "id": "scale",
                    "property": "value"
                }
            ],
        }

    response = HTTP.post(
        f'https://www.lookintobitcoin.com/django_plotly_dash/app/{url_selector}/_dash-update-component',
        json=request_data)
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
