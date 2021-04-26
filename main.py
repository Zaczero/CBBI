import json
from datetime import timedelta
from filecache import filecache
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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

cli_ui.CONFIG['color'] = 'always'

HTTP_TIMEOUT = 30


def mark_highs_lows(df: pd.DataFrame, col: str, begin_with_high: bool, window_size: float, ignore_last_rows: int) -> pd.DataFrame:
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
    col_high = col + 'High'
    col_low = col + 'Low'

    assert col in df.columns, 'The column name (col) could not be found inside the given DataFrame (df)'
    assert col_high not in df.columns, 'The DataFrame (df) already contains the "High" column - bugprone'
    assert col_low not in df.columns, 'The DataFrame (df) already contains the "Low" column - bugprone'
    assert window_size > 0, 'Value of the window_size argument must be at least 1'

    df[col_high] = 0
    df[col_low] = 0

    searching_high = begin_with_high
    current_index = df.index[0]

    while True:
        window = df.loc[current_index:current_index + window_size, col]
        window_index = window.idxmax() if searching_high else window.idxmin()

        if window.shape[0] <= 1:
            break

        if window_index == current_index:
            df.loc[window_index, col_high if searching_high else col_low] = 1
            searching_high = not searching_high
            window_index = window_index + 1

        current_index = window_index

    df.loc[df.shape[0]-ignore_last_rows:, (col_high, col_low)] = 0
    return df


@filecache(3600 * 2)  # 2 hours cache
def fetch_bitcoin_data() -> pd.DataFrame:
    """
    Fetches Bitcoin data into a DataFrame for the past ``past_days`` days.
    The current day is skipped as the data for it is often incomplete (e.g. TotalGenerationUSD).

    Args:
        past_days: Amount of past days to fetch the Bitcoin data of.

    Returns:
        DataFrame containing Bitcoin data.
    """
    cli_ui.info_2(f'Fetching historical Bitcoin data')

    response = requests.get('https://api.blockchair.com/bitcoin/blocks', {
        'a': 'date,count(),min(id),max(id),sum(generation),sum(generation_usd)',
        's': 'date(desc)',
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
    df = df[df['Date'] >= '2011-06-27']

    current_price = df['Price'].tail(1).values[0]
    cli_ui.info_1(f'Current Bitcoin price: ${round(current_price):,}')

    # drop last row (current day)
    return df.iloc[:-1]


@filecache(3600 * 24 * 3)  # 3 day cache
def fetch_google_trends_data(keyword: str, timeframe: str, repeat_if_empty: bool) -> pd.DataFrame:
    pytrends = TrendReq()

    for _ in range(5):
        pytrends.build_payload(kw_list=[keyword], timeframe=timeframe)
        df = pytrends.interest_over_time()

        if df.shape[0] > 0 or not repeat_if_empty:
            return df

        time.sleep(10)

    raise Exception(f'Google Trends API returned no data (timeframe="{timeframe}").')


def add_google_trends_index(df: pd.DataFrame) -> (str, pd.DataFrame):
    """
    Calculates the current "Bitcoin" Google Trends index and appends it to the DataFrame.

    Args:
        df: DataFrame containing Bitcoin data.

    Returns:
        Tuple.
            The first element is a string containing the column name of the index values.
            The second element is a modified source DataFrame containing the index values.

    References:
        Source: https://trends.google.com/trends/explore?date=today%205-y&q=bitcoin
    """

    target_ratio = 1 / 0.2

    cli_ui.info_2(f'Fetching Google Trends data')

    keyword = 'Bitcoin'
    date_start = df.iloc[0]['Date']
    date_end = df.iloc[-1]['Date']
    date_current = date_start
    delta = timedelta(days=90)
    iteration_count = np.ceil((date_end - date_start) / delta).astype(int)

    df_interest = pd.DataFrame()

    cli_ui.info_1('Progress: [', end='')

    for i in range(iteration_count):
        if i % 4 == 0:
            print('#', end='')

        date_start_str = date_current.strftime('%Y-%m-%d')
        date_current = min(date_current + delta, date_end)
        date_end_str = date_current.strftime('%Y-%m-%d')
        timeframe = f'{date_start_str} {date_end_str}'
        repeat_if_empty = i + 1 < iteration_count

        df_fetched = fetch_google_trends_data(keyword, timeframe, repeat_if_empty)

        if df_interest.shape[0] > 0:
            prev_scale = df_interest.iloc[-1][keyword]
            next_scale = df_fetched.iloc[0][keyword]
            ratio_scale = next_scale / prev_scale

            if ratio_scale > 1:
                df_fetched[keyword] /= ratio_scale
            elif ratio_scale < 1:
                df_interest[keyword] *= ratio_scale

            df_fetched = df_fetched.iloc[1:]

        df_interest = df_interest.append(df_fetched)

    print(']')

    df_interest.reset_index(inplace=True)
    df_interest.rename(columns={
        'date': 'Date',
        'Bitcoin': 'Interest'
    }, inplace=True)
    df_interest = mark_highs_lows(df_interest, 'Interest', True, round(365 * 2), 365)

    df = df.join(df_interest.set_index('Date'), on='Date')
    df.fillna({'InterestHigh': 0, 'InterestLow': 0}, inplace=True)
    df['Interest'].ffill(inplace=True)

    def find_previous_high(row: pd.Series):
        peak_indexes = df[df['InterestHigh'] == 1].index
        bin_index = np.digitize(row.name, peak_indexes, right=True)
        peak_value = np.NaN if bin_index == 0 else df.loc[peak_indexes[bin_index - 1], 'Interest']
        return peak_value

    df['PreviousHighInterest'] = df.apply(find_previous_high, axis=1)
    df['GoogleTrendsIndex'] = df['Interest'] / (df['PreviousHighInterest'] * target_ratio)
    return 'GoogleTrendsIndex', df


def add_rupl_index(df: pd.DataFrame) -> (str, pd.DataFrame):
    """
    Calculates the current Relative Unrealized Profit/Loss index and appends it to the DataFrame.

    Args:
        df: DataFrame containing Bitcoin data.

    Returns:
        Tuple.
            The first element is a string containing the column name of the index values.
            The second element is a modified source DataFrame containing the index values.

    References:
        Source: https://www.lookintobitcoin.com/charts/relative-unrealized-profit--loss/
    """

    cli_ui.info_2('Fetching RUPL data')

    response = requests.get('https://www.lookintobitcoin.com/django_plotly_dash/app/unrealised_profit_loss/_dash-layout', timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    response_json = response.json()
    response_x = response_json['props']['children'][0]['props']['figure']['data'][0]['x']
    response_y = response_json['props']['children'][0]['props']['figure']['data'][0]['y']

    df_rupl = pd.DataFrame({
        'Date': response_x[:len(response_y)],
        'RUPL': response_y,
    })
    df_rupl['Date'] = pd.to_datetime(df_rupl['Date']).dt.tz_localize(None)
    df_rupl = mark_highs_lows(df_rupl, 'RUPL', True, round(365 * 2), 365)

    df = df.join(df_rupl.set_index('Date'), on='Date')
    df.fillna({'RUPLHigh': 0, 'RUPLLow': 0}, inplace=True)
    df['RUPL'].ffill(inplace=True)

    low_rows = df.loc[df['RUPLLow'] == 1]
    low_x = low_rows.index.values.reshape(-1, 1)
    low_y = low_rows['RUPL'].values.reshape(-1, 1)

    high_rows = df.loc[df['RUPLHigh'] == 1]
    high_x = high_rows.index.values.reshape(-1, 1)
    high_y = high_rows['RUPL'].values.reshape(-1, 1)

    x = df.index.values.reshape(-1, 1)

    lin_model = LinearRegression()
    lin_model.fit(low_x, low_y)
    df['RUPLLowModel'] = lin_model.predict(x)

    lin_model.fit(high_x, high_y)
    df['RUPLHighModel'] = lin_model.predict(x)

    # sns.set()
    # _, ax = plt.subplots()
    # sns.lineplot(x='Date', y='RUPL', data=df, ax=ax)
    # sns.lineplot(x='Date', y='RUPLHighModel', data=df, ax=ax, color='g')
    # sns.lineplot(x='Date', y='RUPLLowModel', data=df, ax=ax, color='r')
    # plt.legend(['RUPL', 'Best high fit', 'Best low fit'])
    # plt.show()

    df['RUPLIndex'] = (df['RUPL'] - df['RUPLLowModel']) / (df['RUPLHighModel'] - df['RUPLLowModel'])
    return 'RUPLIndex', df


def add_golden_ratio_index(df: pd.DataFrame) -> str:
    """
    Calculates the current Golden Ratio index and appends it to the DataFrame inline.

    Args:
        df: DataFrame containing Bitcoin data.

    Returns:
        String containing the column name of the index values.

    References:
        Source: https://www.tradingview.com/chart/BTCUSD/QBeNL8jt-BITCOIN-The-Golden-51-49-Ratio-600-days-of-Bull-Market-left/
    """

    current_bottom_date = pd.to_datetime('2018-12-03')
    current_halving_date = pd.to_datetime('2020-05-11')
    current_peak_date = current_bottom_date + (current_halving_date - current_bottom_date) / .51

    df['GoldenRatioIndex'] = (df['Date'] - current_bottom_date) / (current_peak_date - current_bottom_date)
    df.loc[df['GoldenRatioIndex'] < 0.51, 'GoldenRatioIndex'] = np.nan

    return 'GoldenRatioIndex'


def add_stock_to_flow_index(df: pd.DataFrame) -> str:
    """
    Calculates the current Stock to Flow index and appends it to the DataFrame inline.

    Args:
        df: DataFrame containing Bitcoin data.

    Returns:
        String containing the column name of the index values.

    References:
        Source: https://digitalik.net/btc/
    """

    previous_hinge_date = pd.to_datetime('2017-10-15')
    previous_peak_date = pd.to_datetime('2017-12-15')
    current_hinge_date = pd.to_datetime('2021-08-15')
    current_peak_date = current_hinge_date + (previous_peak_date - previous_hinge_date)

    df['StockToFlowIndex'] = (df['Date'] - previous_peak_date) / (current_peak_date - previous_peak_date)
    df.loc[df['StockToFlowIndex'] < 0, 'StockToFlowIndex'] = np.nan

    return 'StockToFlowIndex'


def add_pi_cycle_index(df: pd.DataFrame) -> str:
    """
    Calculates the current Pi Cycle index and appends it to the DataFrame inline.

    Args:
        df: DataFrame containing Bitcoin data.

    Returns:
        String containing the column name of the index values.

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

    df['PiCycleIndex'] = 1 - (df['PiCycleDifference'] / max_divergence)

    return 'PiCycleIndex'


def add_2yma_index(df: pd.DataFrame) -> str:
    """
    Calculates the current 2-Year Moving Average index and appends it to the DataFrame inline.

    Args:
        df: DataFrame containing Bitcoin data.

    Returns:
        String containing the column name of the index values.

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

    df['2YMAIndex'] = (df['PriceLog'] - df['2YMALogUndershoot']) / (df['2YMALogOvershoot'] - df['2YMALogUndershoot'])

    return '2YMAIndex'


def add_trolololo_index(df: pd.DataFrame) -> str:
    """
    Calculates the current Trolololo index and appends it to the DataFrame inline.

    Args:
        df: DataFrame containing Bitcoin data.

    Returns:
        String containing the column name of the index values.

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

    df['TrolololoIndex'] = (df['PriceLog'] - df['TrolololoLogBottom']) / (df['TrolololoLogTop'] - df['TrolololoLogBottom'])

    return 'TrolololoIndex'


def add_puell_index(df: pd.DataFrame) -> str:
    """
    Calculates the current Puell Multiple index and appends it to the DataFrame inline.

    Args:
        df: DataFrame containing Bitcoin data.

    Returns:
        String containing the column name of the index values.

    References:
        Source: https://www.lookintobitcoin.com/charts/puell-multiple/
    """

    projected_max = np.log(4.75)
    projected_min = np.log(0.3)

    df['PuellMA365'] = df['TotalGenerationUSD'].rolling(365).mean()
    df['Puell'] = df['TotalGenerationUSD'] / df['PuellMA365']
    df['PuellLog'] = np.log(df['Puell'])

    df['PuellIndex'] = (df['PuellLog'] - projected_min) / (projected_max - projected_min)

    return 'PuellIndex'


def format_percentage(val: float, suffix: str = ' %') -> str:
    """
    Formats a percentage value (0.0 - 1.0) in a standardized way.
    Returned value has a constant width and a trailing '%' sign.

    Args:
        val: Percentage value to be formatted.
        suffix: String to be appended to the result.

    Returns:
        Formatted percentage value with a constant width and trailing '%' sign.

    Examples:
        >>> print(format_percentage(0.359))
        str(' 36 %')

        >>> print(format_percentage(1.1))
        str('110 %')
    """

    return f'{round(val * 100): >3d}{suffix}'


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

    df_bitcoin = fetch_bitcoin_data()

    col_google_trends, df_bitcoin = add_google_trends_index(df_bitcoin)
    col_rupl, df_bitcoin = add_rupl_index(df_bitcoin)

    # parse and analyse the data
    col_golden_ratio = add_golden_ratio_index(df_bitcoin)
    col_stock_to_flow = add_stock_to_flow_index(df_bitcoin)
    col_pi_cycle = add_pi_cycle_index(df_bitcoin)
    col_2yma = add_2yma_index(df_bitcoin)
    col_trolololo = add_trolololo_index(df_bitcoin)
    col_puell = add_puell_index(df_bitcoin)
    col_confidence = 'Confidence'

    cols_metric = [
        col_google_trends,
        col_rupl,
        col_golden_ratio,
        col_stock_to_flow,
        col_pi_cycle,
        col_2yma,
        col_trolololo,
        col_puell,
    ]

    df_result = pd.DataFrame(df_bitcoin[['Date', 'Price'] + cols_metric])

    df_result.dropna(inplace=True)
    df_result.set_index('Date', inplace=True)
    df_result[col_confidence] = df_result[cols_metric].mean(axis=1)
    df_result.to_json(file,
                      double_precision=4,
                      date_unit='s',
                      indent=2,
                      )

    df_result_latest = df_result.tail(1)

    details_stdout = {
        'The Golden 51%-49% Ratio': df_result_latest[col_golden_ratio][0],
        '"Bitcoin" search term (Google Trends)': df_result_latest[col_google_trends][0],
        'Stock-to-Flow Chart': df_result_latest[col_stock_to_flow][0],
        'Pi Cycle Top Indicator': df_result_latest[col_pi_cycle][0],
        '2 Year Moving Average': df_result_latest[col_2yma][0],
        'Bitcoin Trolololo Trend Line': df_result_latest[col_trolololo][0],
        'RUPL/NUPL Chart': df_result_latest[col_rupl][0],
        'Puell Multiple': df_result_latest[col_puell][0],
    }

    print('\n')
    cli_ui.info_3('Confidence we are at the peak:')
    cprint(figlet_format(format_percentage(df_result_latest[col_confidence][0], ''), font='univers'), 'cyan', attrs=['bold'], end='')

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
