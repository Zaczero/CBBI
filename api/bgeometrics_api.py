import pandas as pd
from filecache import filecache

from utils import HTTP


@filecache(7200)  # Cache for 2 hours to avoid rate limiting
def bg_fetch(endpoint: str, value_col: str, col_name: str) -> pd.DataFrame:
    """
    Fetch data from BGeometrics free API.
    
    Args:
        endpoint: The API endpoint (e.g., 'nupl', 'mvrv', 'rhodl-ratio', 'reserve-risk')
        value_col: The column name in the JSON response containing the value
        col_name: The column name to use in the output DataFrame
    
    Returns:
        DataFrame with 'Date' and col_name columns
    """
    url = f'https://bitcoin-data.com/api/v1/{endpoint}'
    print(f'🔍 Fetching from BGeometrics API: {url}')
    
    response = HTTP.get(url)
    print(f'📡 Response status: {response.status_code}')
    response.raise_for_status()
    
    data = response.json()
    print(f'📊 Received {len(data)} data points from BGeometrics')
    
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['unixTs'].astype(int), unit='s').dt.tz_localize(None)
    df[col_name] = df[value_col].astype(float)
    
    print(f'✅ Successfully fetched {len(df)} rows for {col_name}')
    return df[['Date', col_name]]
