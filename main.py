import json
import fire
import numpy as np
import pandas as pd
import requests
import time
import traceback
from pytrends.request import TrendReq
from pyfiglet import figlet_format
from termcolor import cprint
import cli_ui

cli_ui.CONFIG['color'] = 'always'

HTTP_TIMEOUT = 30


def mark_highs_lows(df: pd.DataFrame, col: str, begin_with_high: bool, window_size: float) -> pd.DataFrame:
    """
    Marks highs and lows (peaks) of the column values inside the given DataFrame.
    Marked points are indicated by the value '1' inside their corresponding, newly added, 'High' and 'Low' columns.

    Args:
        df: DataFrame from which the column values are selected and to which marked points columns are added.
        col: Column name of which values are selected inside the given DataFrame.
        begin_with_high: Indicates whether the first peak is high or low.
        window_size: Window size for the algorithm to consider.
                     Too low value will mark too many peaks, whereas, too high value will mark too little peaks.

    Returns:
        Modified input DataFrame with columns, indicating the marked points, added.
    """
    assert col in df.columns, 'The column name (col) could not be found inside the given DataFrame (df)'
    assert window_size > 0, 'Value of the window_size argument must be at least 1'
    assert 'High' not in df.columns, 'The DataFrame (df) already contains the "High" column - bugprone'
    assert 'Low' not in df.columns, 'The DataFrame (df) already contains the "Low" column - bugprone'

    df['High'] = 0
    df['Low'] = 0

    searching_high = begin_with_high
    current_index = df.index[0]

    while True:
        window = df.loc[current_index:current_index + window_size, col]
        window_index = window.idxmax() if searching_high else window.idxmin()

        if window.shape[0] <= 1:
            break

        if window_index == current_index:
            df.loc[window_index, 'High' if searching_high else 'Low'] = 1
            searching_high = not searching_high
            window_index = window_index + 1

        current_index = window_index

    return df


def get_google_trends_index() -> float:
    """
    Calculates the current "Bitcoin" Google Trends index.

    Returns:
        Percentage value (0.0 - 1.0) of the "Bitcoin" Google Trends index.

    References:
        Source: https://trends.google.com/trends/explore?date=today%205-y&q=bitcoin
    """

    target_ratio = 1 / .125

    cli_ui.info_2('Fetching Google Trends data')

    pytrends = TrendReq()
    pytrends.build_payload(kw_list=['Bitcoin'])
    df_interest = pytrends.interest_over_time()

    if df_interest.shape[0] < 100:
        raise Exception('Google Trends returned too little data.')

    df_interest.reset_index(inplace=True)
    df_interest.rename(columns={
        'date': 'Date',
        'Bitcoin': 'Interest'
    }, inplace=True)
    df_interest = mark_highs_lows(df_interest, 'Interest', True, round(365 * 2 / 7))

    previous_peak = df_interest.loc[df_interest['High'] == 1].head(1)['Interest'].values[0]
    current_peak = df_interest.tail(1)['Interest'].values[0]

    cli_ui.info_1(f'Previous Google Trends peak: {previous_peak}%')
    cli_ui.info_1(f'Current Google Trends peak: {current_peak}%')

    current_ratio = current_peak / previous_peak
    return current_ratio / target_ratio


def get_rupl_index() -> float:
    """
    Calculates the current Relative Unrealized Profit/Loss index.

    Returns:
        Percentage value (0.0 - 1.0) of the Relative Unrealized Profit/Loss index.

    References:
        Source: https://www.lookintobitcoin.com/charts/relative-unrealized-profit--loss/
    """

    projected_max = .75
    projected_min = -.2

    cli_ui.info_2('Fetching RUPL data')

    response = requests.get('https://www.lookintobitcoin.com/django_plotly_dash/app/unrealised_profit_loss/_dash-layout', timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    response_json = response.json()

    current_value = response_json['props']['children'][0]['props']['figure']['data'][0]['y'][-1]

    return (current_value - projected_min) / (projected_max - projected_min)


def fetch_bitcoin_data(past_days: int) -> pd.DataFrame:
    """
    Fetches Bitcoin data into a DataFrame for the past ``past_days`` days.
    The current day is skipped as the data for it is often incomplete (e.g. TotalGenerationUSD).

    Args:
        past_days: Amount of past days to fetch the Bitcoin data of.

    Returns:
        DataFrame containing Bitcoin data.
    """
    assert past_days > 0, 'Value of the past_days argument must be at least 1'

    cli_ui.info_2(f'Fetching Bitcoin data for the past {past_days} days')

    response = requests.get('https://api.blockchair.com/bitcoin/blocks', {
        'a': 'date,count(),min(id),max(id),sum(generation),sum(generation_usd)',
        's': 'date(desc)',
        'limit': past_days + 1,
    }, timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    response_json = response.json()

    df = pd.DataFrame(response_json['data'][::-1])
    df.rename(columns={
        'date': 'Date',
        'count()': 'TotalBlocks',
        'min(id)': 'MinBlockID',
        'max(id)': 'MaxBlockID',
        'sum(generation)': 'TotalGeneration',
        'sum(generation_usd)': 'TotalGenerationUSD'
    }, inplace=True)

    df['Date'] = pd.to_datetime(df['Date'])
    df['TotalGeneration'] /= 1e8
    df['BlockGeneration'] = df['TotalGeneration'] / df['TotalBlocks']
    df['BlockGenerationUSD'] = df['TotalGenerationUSD'] / df['TotalBlocks']
    df['Price'] = df['BlockGenerationUSD'] / df['BlockGeneration']
    df['PriceLog'] = np.log(df['Price'])

    current_price = df['Price'].tail(1).values[0]
    cli_ui.info_1(f'Current Bitcoin price: ${round(current_price):,}')

    return df.head(past_days)


def get_golden_ratio_index(df: pd.DataFrame) -> float:
    """
    Calculates the current Golden Ratio index.

    Args:
        df: DataFrame containing Bitcoin data.

    Returns:
        Percentage value (0.0 - 1.0) of the Golden Ratio index.

    References:
        Source: https://www.tradingview.com/chart/BTCUSD/QBeNL8jt-BITCOIN-The-Golden-51-49-Ratio-600-days-of-Bull-Market-left/
    """

    current_bottom_date = pd.to_datetime('2018-12-03')
    current_halving_date = pd.to_datetime('2020-05-11')
    current_peak_date = current_bottom_date + (current_halving_date - current_bottom_date) / .51
    current_date = df['Date'].tail(1).values[0]

    return (current_date - current_bottom_date) / (current_peak_date - current_bottom_date)


def get_sf_index(df: pd.DataFrame) -> float:
    """
    Calculates the current Stock to Flow index.

    Args:
        df: DataFrame containing Bitcoin data.

    Returns:
        Percentage value (0.0 - 1.0) of the Stock to Flow index.

    References:
        Source: https://digitalik.net/btc/
    """

    previous_hinge_date = pd.to_datetime('2017-10-15')
    previous_peak_date = pd.to_datetime('2017-12-15')
    current_hinge_date = pd.to_datetime('2021-08-15')
    current_peak_date = current_hinge_date + (previous_peak_date - previous_hinge_date)
    current_date = df['Date'].tail(1).values[0]

    return (current_date - previous_peak_date) / (current_peak_date - previous_peak_date)


def get_pi_cycle_index(df: pd.DataFrame) -> float:
    """
    Calculates the current Pi Cycle index.

    Args:
        df: DataFrame containing Bitcoin data.

    Returns:
        Percentage value (0.0 - 1.0) of the Pi Cycle index.

    References:
        Source: https://www.lookintobitcoin.com/charts/pi-cycle-top-indicator/
    """

    max_divergence_price_high = np.log(12000)
    max_divergence_price_low = np.log(3700)
    max_divergence = max_divergence_price_high - max_divergence_price_low

    df['111DMA'] = df['Price'].rolling(111).mean()
    df['350DMAx2'] = df['Price'].rolling(350).mean() * 2

    df['111DMALog'] = np.log(df['111DMA'])
    df['350DMAx2Log'] = np.log(df['350DMAx2'])
    df['PiCycleDifference'] = np.abs(df['111DMALog'] - df['350DMAx2Log'])

    return (1 - df['PiCycleDifference'] / max_divergence).tail(1).values[0]


def get_2yma_index(df: pd.DataFrame) -> float:
    """
    Calculates the current 2-Year Moving Average index.

    Args:
        df: DataFrame containing Bitcoin data.

    Returns:
        Percentage value (0.0 - 1.0) of the 2-Year Moving Average index.

    References:
        Source: https://www.lookintobitcoin.com/charts/bitcoin-investor-tool/
    """

    overshoot = .2

    df['2YMA'] = df['Price'].rolling(365 * 2).mean()
    df['2YMAx5'] = df['2YMA'] * 5

    df['2YMALog'] = np.log(df['2YMA'])
    df['2YMAx5Log'] = np.log(df['2YMAx5'])
    df['2YMALogDifference'] = df['2YMAx5Log'] - df['2YMALog']
    df['2YMALogOvershootDifference'] = df['2YMALogDifference'] * overshoot
    df['2YMALogOvershoot'] = df['2YMAx5Log'] + df['2YMALogOvershootDifference']
    df['2YMALogUndershoot'] = df['2YMALog'] - df['2YMALogOvershootDifference']

    return ((df['PriceLog'] - df['2YMALogUndershoot']) / (df['2YMALogOvershoot'] - df['2YMALogUndershoot'])).tail(1).values[0]


def get_trolololo_index(df: pd.DataFrame) -> float:
    """
    Calculates the current Trolololo index.

    Args:
        df: DataFrame containing Bitcoin data.

    Returns:
        Percentage value (0.0 - 1.0) of the Trolololo index.

    References:
        Source: https://www.blockchaincenter.net/bitcoin-rainbow-chart/
    """

    begin_date = pd.to_datetime('2012-01-01')
    log_diff_line_count = 8
    log_diff_top = 7
    log_diff_bottom = 0.5

    df['TrolololoDaysSinceBegin'] = (df['Date'] - begin_date).dt.days

    df['TrolololoLineTopPrice'] = np.power(10, 2.900 * np.log(df['TrolololoDaysSinceBegin'] + 1400) - 19.463)  # Maximum Bubble Territory
    df['TrolololoLineTopPriceLog'] = np.log(df['TrolololoLineTopPrice'])
    df['TrolololoLineBottomPrice'] = np.power(10, 2.788 * np.log(df['TrolololoDaysSinceBegin'] + 1200) - 19.463)  # Basically a Fire Sale
    df['TrolololoLineBottomPriceLog'] = np.log(df['TrolololoLineBottomPrice'])

    df['TrolololoLogDifference'] = (df['TrolololoLineTopPriceLog'] - df['TrolololoLineBottomPriceLog']) / log_diff_line_count
    df['TrolololoLogTop'] = df['TrolololoLineBottomPriceLog'] + log_diff_top * df['TrolololoLogDifference']
    df['TrolololoLogBottom'] = df['TrolololoLineBottomPriceLog'] - log_diff_bottom * df['TrolololoLogDifference']

    return ((df['PriceLog'] - df['TrolololoLogBottom']) / (df['TrolololoLogTop'] - df['TrolololoLogBottom'])).tail(1).values[0]


def get_puell_index(df: pd.DataFrame) -> float:
    """
    Calculates the current Puell Multiple index.

    Args:
        df: DataFrame containing Bitcoin data.

    Returns:
        Percentage value (0.0 - 1.0) of the Puell Multiple index.

    References:
        Source: https://www.lookintobitcoin.com/charts/puell-multiple/
    """

    projected_max = np.log(4.75)
    projected_min = np.log(0.3)

    df['PuellMA365'] = df['TotalGenerationUSD'].rolling(365).mean()
    df['Puell'] = df['TotalGenerationUSD'] / df['PuellMA365']
    df['PuellLog'] = np.log(df['Puell'])

    return ((df['PuellLog'] - projected_min) / (projected_max - projected_min)).tail(1).values[0]


def format_percentage(val: float) -> str:
    """
    Formats a percentage value (0.0 - 1.0) in a standardized way.
    Returned value has a constant width and a trailing '%' sign.

    Args:
        val: Percentage value to be formatted.

    Returns:
        Formatted percentage value with a constant width and trailing '%' sign.

    Examples:
        >>> print(format_percentage(0.359))
        str(' 36 %')

        >>> print(format_percentage(1.1))
        str('110 %')
    """

    return f'{round(val * 100): >3d} %'


def get_color(val: float) -> str:
    """
    Maps a percentage value (0.0 - 1.0) to its corresponding color.
    The color is used to indicate whether the value is low (0.0) or high (1.0).
    Returned value is a valid termcolor-package color string.

    Args:
        val: Percentage value to be mapped into a color.

    Returns:
        Valid termcolor-package color string.
    """

    config = [
        'red',
        .3,
        'yellow',
        .65,
        'green',
        .85,
        'cyan',
        .97,
        'magenta',
    ]

    bin_index = np.digitize([round(val, 2)], config[1::2])[0]
    return config[::2][bin_index]


def run(file: str) -> None:
    """
    Calculates the current CBBI confidence value alongside all the required metrics.
    Everything gets pretty printed to the current stdout and a clean copy
    is written to a JSON file specified by the path in the ``file`` argument.

    Args:
        file: File path where the output is stored in the JSON format.

    Returns:
        None
    """

    # fetch online data
    google_trends_index = get_google_trends_index()
    rupl_index = get_rupl_index()
    df_bitcoin = fetch_bitcoin_data(365 * 2)

    # parse and analyse the data
    golden_ratio_index = get_golden_ratio_index(df_bitcoin)
    sf_index = get_sf_index(df_bitcoin)
    pi_cycle_index = get_pi_cycle_index(df_bitcoin)
    _2yma_index = get_2yma_index(df_bitcoin)
    trolololo_index = get_trolololo_index(df_bitcoin)
    puell_index = get_puell_index(df_bitcoin)

    confidence = np.mean([
        golden_ratio_index,
        google_trends_index,
        sf_index,
        pi_cycle_index,
        _2yma_index,
        trolololo_index,
        rupl_index,
        puell_index
    ])

    details_json = {
        'confidence': confidence,
        'golden_ratio': golden_ratio_index,
        'google_trends': google_trends_index,
        'stock_to_flow': sf_index,
        'pi_cycle': pi_cycle_index,
        '2yma': _2yma_index,
        'trolololo': trolololo_index,
        'rupl': rupl_index,
        'puell': puell_index,
        'timestamp': int(time.time())
    }

    with open(file, 'w+') as f:
        json.dump(details_json, f, indent=2)

    details_stdout = {
        'The Golden 51%-49% Ratio': golden_ratio_index,
        '"Bitcoin" search term (Google Trends)': google_trends_index,
        'Stock-to-Flow Chart': sf_index,
        'Pi Cycle Top Indicator': pi_cycle_index,
        '2 Year Moving Average': _2yma_index,
        'Bitcoin Trolololo Trend Line': trolololo_index,
        'RUPL/NUPL Chart': rupl_index,
        'Puell Multiple': puell_index,
    }

    print('\n')
    cli_ui.info_3('Confidence we are at the peak:')
    cprint(figlet_format(format_percentage(confidence), font='univers'), 'cyan', attrs=['bold'], end='')

    for k, v in details_stdout.items():
        cprint(format_percentage(v) + ' ', color=get_color(v), attrs=['reverse'], end='')
        print(f' - {k}')

    print()
    cli_ui.info_3('Source code: https://github.com/Zaczero/CBBI', end='\n\n')


def run_and_retry(file: str, max_attempts: int = 10, sleep_seconds_on_error: float = 10) -> None:
    """
    Calculates the current CBBI confidence value alongside all the required metrics.
    Everything gets pretty printed to the current stdout and a clean copy
    is written to a JSON file specified by the path in the ``file`` argument.
    The execution will be attempted multiple times in case an error occurs.

    Args:
        file: File path where the output is stored in the JSON format.
        max_attempts: Maximum number of attempts before termination. An attempt is counted when an error occurs.
        sleep_seconds_on_error: Duration of the sleep in seconds before attempting again after an error occurs.

    Returns:
        None
    """
    assert max_attempts > 0, 'Value of the max_attempts argument must be at least 1'
    assert sleep_seconds_on_error >= 0, 'Value of the sleep_seconds_on_error argument must be positive'

    for _ in range(max_attempts):
        try:
            run(file)
            exit(0)

        except Exception:
            cli_ui.error('An error occurred!')
            traceback.print_exc()

            print()
            cli_ui.info_1(f'Retrying in {sleep_seconds_on_error} seconds...')
            time.sleep(sleep_seconds_on_error)

    cli_ui.info_1(f'Max attempts limit has been reached ({max_attempts}). Better luck next time!')
    exit(-1)


if __name__ == '__main__':
    fire.Fire(run_and_retry)
