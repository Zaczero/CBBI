import os
import traceback
from math import ceil

import numpy as np
import polars as pl
import telegram
from httpx import Client
from sty import bg

HTTP = Client(
    headers={
        'User-Agent': 'Mozilla/5.0 (Linux x86_64; rv:140.0) Gecko/20100101 Firefox/140.0'
    },
    timeout=60,
    follow_redirects=True,
)


def mark_highs_lows(
    df: pl.DataFrame,
    col: str,
    begin_with_high: bool,
    window_size: int,
    ignore_last_rows: int,
):
    """
    Marks highs and lows (peaks) of the column values inside the given DataFrame.
    Marked points are indicated by `True` inside their corresponding, newly added, `col + "High"` and `col + "Low"` columns.

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

    values = df.get_column(col).to_numpy()
    high_marks = np.zeros(len(values), dtype=np.bool_)
    low_marks = np.zeros(len(values), dtype=np.bool_)

    searching_high = begin_with_high
    current_index = 0

    while True:
        window_end = min(current_index + window_size, len(values) - 1)
        window = values[current_index : window_end + 1]

        if sum(~np.isnan(window)) == 0 and window.shape[0] > 1:
            current_index += window.shape[0]
            continue

        if window.shape[0] <= 1:
            break

        window_index = current_index + (
            np.nanargmax(window) if searching_high else np.nanargmin(window)
        )

        if window_index == current_index:
            if searching_high:
                high_marks[window_index] = True
            else:
                low_marks[window_index] = True
            searching_high = not searching_high
            window_index = window_index + 1

        current_index = window_index

    if ignore_last_rows > 0:
        high_marks[-ignore_last_rows:] = False
        low_marks[-ignore_last_rows:] = False

    # stabilize the algorithm until a next major update
    stabilize_mask = df.get_column('Date').to_numpy() >= np.datetime64('2023-07-01')
    high_marks[stabilize_mask] = False
    low_marks[stabilize_mask] = False

    return df.with_columns(
        pl.Series(col_high, high_marks, dtype=pl.Boolean),
        pl.Series(col_low, low_marks, dtype=pl.Boolean),
    )


def mark_days_since(df: pl.DataFrame, cols: list[str]):
    """
    This function takes a DataFrame and a list of column names
    and calculates the number of days since the last `True` value for each column in the list.

    The resulting DataFrame will have new columns for each input column, with the column name prefixed by 'DaysSince'.
    The value in these new columns will be the number of days since the last `True` value in the corresponding input column.

    Args:
        df: The input DataFrame.
        cols: The list of boolean marker columns to calculate the days since the last `True` value.

    Returns:
        The modified DataFrame with the new columns added.
    """
    df = df.with_row_index(name='_row_nr')

    exprs: dict[str, pl.Expr] = {}
    for col in cols:
        last_event_idx = (
            pl.when(pl.col(col)).then(pl.col('_row_nr')).otherwise(None).forward_fill()
        )
        exprs[f'DaysSince{col}'] = pl.col('_row_nr') - last_event_idx

    return df.with_columns(**exprs).drop('_row_nr')


def format_percentage(val: float, suffix: str = ' %'):
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


def get_color(val: float):
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


async def send_error_notification(exception: Exception):
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
