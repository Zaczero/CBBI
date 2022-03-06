import pandas as pd
import requests

from globals import *


def cbbi_fetch(key: str) -> pd.DataFrame:
    response = requests.get(
        'https://colintalkscrypto.com/cbbi/data/latest.json',
        headers={'User-Agent': USER_AGENT},
        timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    response_data = response.json()[key]

    df = pd.DataFrame(response_data.items(), columns=[
        'Date',
        'Value',
    ])
    df['Date'] = pd.to_datetime(df['Date'], unit='s').dt.tz_localize(None)

    return df
