import os
import traceback
from datetime import datetime
from math import ceil

import numpy as np
import pandas as pd
import seaborn as sns
import telegram
from httpx import Client
from matplotlib.axes import Axes
from sty import bg

HTTP = Client(
    headers={'User-Agent': 'Mozilla/5.0 (Linux x86_64; rv:140.0) Gecko/20100101 Firefox/140.0'},
    timeout=30,
    follow_redirects=True,
)


def mark_highs_lows(
    df: pd.DataFrame,
    col: str,
    begin_with_high: bool,
    window_size: float,
    ignore_last_rows: int,
) -> pd.DataFrame:
    """
    Marks highs and lows (peaks) of the column values inside the given DataFrame.
    Marked points are indicated by the value '1' inside their corresponding, newly added, '``col``High' and '``col``Low' columns.

    Args:
        df: DataFrame from which the column values are selected and to which marked points columns are added.
        col: Column name of which values are selected inside the given DataFrame.
        begin_with_high: Indicates whether the first peak is high or low.
        window_size: Window size for the algorithm to consider.
                     Too low value will mark too many peaks, whereas, too high value will mark too little peaks.
        ignore_last_rows: Amount of trailing DataFrame rows for which highs and lows should not be marked.

    Returns:
        Modified input DataFrame with columns, indicating the marked points, added.
    """
    col_high = col + 'High'
    col_low = col + 'Low'

    assert col in df.columns, f'The column name "{col}" (col) could not be found inside the given DataFrame (df)'
    assert col_high not in df.columns, 'The DataFrame (df) already contains the "High" column - bug prone'
    assert col_low not in df.columns, 'The DataFrame (df) already contains the "Low" column - bug prone'
    assert window_size > 0, 'Value of the window_size argument must be at least 1'

    df[col_high] = 0
    df[col_low] = 0

    searching_high = begin_with_high
    current_index = df.index[0]

    while True:
        window = df.loc[current_index : current_index + window_size, col]

        if sum(~np.isnan(window)) == 0 and window.shape[0] > 1:
            current_index += window.shape[0]
            continue

        if window.shape[0] <= 1:
            break

        window_index = window.idxmax() if searching_high else window.idxmin()

        if window_index == current_index:
            df.loc[window_index, col_high if searching_high else col_low] = 1
            searching_high = not searching_high
            window_index = window_index + 1

        current_index = window_index

    df.loc[df.shape[0] - ignore_last_rows :, (col_high, col_low)] = 0

    # stabilize the algorithm until a next major update
    df.loc[df['Date'] >= '2023-07-01', (col_high, col_low)] = 0
    return df


def mark_days_since(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    This function takes a DataFrame and a list of column names
    and calculates the number of days since the last value of 1 for each column in the list.

    The resulting DataFrame will have new columns for each input column, with the column name prefixed by 'DaysSince'.
    The value in these new columns will be the number of days since the last value of 1 in the corresponding input column.

    Args:
        df: The input DataFrame.
        cols: The list of columns in the DataFrame to calculate the days since the last value of 1.

    Returns:
        The modified DataFrame with the new columns added.
    """
    for col in cols:
        indexes = df.loc[df[col] == 1].index
        df[f'DaysSince{col}'] = df.index.to_series().apply(
            lambda v: min([v - index if index <= v else np.nan for index in indexes])  # noqa: B023
        )

    return df


def add_common_markers(df: pd.DataFrame, ax: Axes, price_line: bool = True) -> None:
    """
    This function adds common markers to a plot.

    Args:
        df: The DataFrame containing the data to be plotted.
        ax: The Axes object to be plotted on.
        price_line: If True, a line plot of the 'PriceLogInterp' column will be added to the Axes. Default is True.

    Returns:
        None
    """
    if price_line:
        sns.lineplot(data=df, x='Date', y='PriceLogInterp', alpha=0.4, color='orange', ax=ax)

    for _, row in df[df['Halving'] == 1].iterrows():
        days_since_epoch = (row['Date'] - datetime(1970, 1, 1)).days
        ax.axvline(x=days_since_epoch, color='navy', linestyle=':')

    for _, row in df[df['PriceHigh'] == 1].iterrows():
        days_since_epoch = (row['Date'] - datetime(1970, 1, 1)).days
        ax.axvline(x=days_since_epoch, color='green', linestyle=':')

    for _, row in df[df['PriceLow'] == 1].iterrows():
        days_since_epoch = (row['Date'] - datetime(1970, 1, 1)).days
        ax.axvline(x=days_since_epoch, color='red', linestyle=':')


def split_df_on_index_gap(df: pd.DataFrame, min_gap: int = 1):
    """
    Split a Pandas DataFrame on gaps in the index values.

    Args:
        df: The DataFrame to split.
        min_gap: The minimum gap size in the index values to split on.

    Returns:
        A list of DataFrames split on the specified gaps in the index values.
    """
    begin_idx = None
    end_idx = None

    for i, _ in df.iterrows():
        if begin_idx is None:
            begin_idx = i
            end_idx = i
        elif (i - end_idx) <= min_gap:
            end_idx = i
        else:
            yield df.loc[begin_idx:end_idx]
            begin_idx = i
            end_idx = i

    if begin_idx is not None:
        yield df.loc[begin_idx:end_idx]


def format_percentage(val: float, suffix: str = ' %') -> str:
    """
    Formats a percentage value (0.0 - 1.0) in the standardized way.
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

    return f'{ceil(val * 100): >3d}{suffix}'


def get_color(val: float) -> str:
    """
    Maps a percentage value (0.0 - 1.0) to its corresponding color.
    The color is used to indicate whether the value is low (0.0) or high (1.0).
    Returned value is a valid sty-package color string.

    Args:
        val: Percentage value to be mapped into a color.

    Returns:
        Valid sty-package color string.
    """

    config = [
        bg.da_red,
        0.3,
        bg.da_yellow,
        0.65,
        bg.da_green,
        0.85,
        bg.da_cyan,
        0.97,
        bg.da_magenta,
    ]

    bin_index = np.digitize([round(val, 2)], config[1::2])[0]
    return config[::2][bin_index]


async def send_error_notification(exception: Exception) -> bool:
    """
    This function sends a notification to a Telegram chat with details of the provided exception.

    Args:
        exception: The exception to be reported.

    Returns:
        A boolean indicating whether the notification was sent successfully.
    """
    telegram_token = os.getenv('TELEGRAM_TOKEN')
    telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
    if not telegram_token or not telegram_chat_id:
        return False

    async with telegram.Bot(telegram_token) as bot:
        await bot.send_message(
            telegram_chat_id,
            f'🚨 An error has occurred: <b>{exception!s}</b>\n'
            f'\n'
            f'🔍️ <b>Stack trace</b>\n'
            f'<pre>{"".join(traceback.format_exception(exception))}</pre>',
            parse_mode='HTML',
        )
    return True
