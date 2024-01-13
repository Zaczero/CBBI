import os

import pandas as pd

from utils import HTTP


def gn_fetch(url_selector: str, col_name: str, **kwargs) -> pd.DataFrame:
    api_key = os.getenv('GLASSNODE_API_KEY')

    if not api_key:
        raise Exception('GlassNode fallback in unavailable (missing api key)')

    response = HTTP.get(
        f'https://api.glassnode.com/v1/metrics/indicators/{url_selector}',
        params=kwargs,
        headers={'X-Api-Key': api_key},
    )
    response.raise_for_status()
    response_json = response.json()
    response_x = [d['t'] for d in response_json]
    response_y = [d['v'] for d in response_json]

    df = pd.DataFrame(
        {
            'Date': response_x,
            col_name: response_y,
        }
    )
    df['Date'] = pd.to_datetime(df['Date'], unit='s').dt.tz_localize(None)

    return df
