import pandas as pd

from utils import HTTP


def cbbi_fetch(key: str) -> pd.DataFrame:
    response = HTTP.get('https://colintalkscrypto.com/cbbi/data/latest.json')
    response.raise_for_status()
    response_data = response.json()[key]

    df = pd.DataFrame(
        response_data.items(),
        columns=[
            'Date',
            'Value',
        ],
    )
    df['Date'] = pd.to_datetime(df['Date'], unit='s').dt.tz_localize(None)

    return df
