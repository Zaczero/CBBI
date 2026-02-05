import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes

from metrics._common import join_left_on_date, linreg_predict
from metrics.base_metric import BaseMetric
from utils import HTTP


def _woocharts_xy_ms_df(*, x: list[object], y: list[object], y_name: str):
    return (
        pl
        .DataFrame({
            'Date': pl.Series(x, dtype=pl.Int64),
            y_name: pl.Series(y, dtype=pl.Float64),
        })
        .with_columns(
            Date=(
                pl
                .from_epoch(pl.col('Date'), time_unit='ms')
                .dt.cast_time_unit('us')
                .dt.replace_time_zone('UTC')
            )
        )
        .select('Date', y_name)
    )


def _fetch_df():
    response = HTTP.get('https://woocharts.com/bitcoin-price-models/data/chart.json')
    response.raise_for_status()
    data = response.json()

    df_top = _woocharts_xy_ms_df(
        x=data['top_']['x'],
        y=data['top_']['y'],
        y_name='Top',
    )
    df_cvdd = _woocharts_xy_ms_df(
        x=data['cvdd']['x'],
        y=data['cvdd']['y'],
        y_name='CVDD',
    )

    return df_top.join(df_cvdd, on='Date', how='inner', maintain_order='left')


class WoobullMetric(BaseMetric):
    @property
    def name(self):
        return 'Woobull'

    @property
    def description(self):
        return 'Woobull Top Cap vs CVDD'

    def _calculate(self, df: pl.DataFrame, ax: list[Axes]):
        df = join_left_on_date(df, _fetch_df())

        row_nr = np.arange(df.height)
        high_idx = row_nr[df.get_column('PriceHigh').to_numpy()]
        low_idx = row_nr[df.get_column('PriceLow').to_numpy()][1:]

        top_log = np.log(df.get_column('Top').to_numpy())
        cvdd_log = np.log(df.get_column('CVDD').to_numpy())
        price_log = df.get_column('PriceLog').to_numpy()
        woobull = (price_log - cvdd_log) / (top_log - cvdd_log)

        high_model = linreg_predict(high_idx, woobull[high_idx], row_nr) - 0.025
        low_model = linreg_predict(low_idx, woobull[low_idx], row_nr)

        x = df.get_column('Date').to_numpy()
        woobull_index = (woobull - low_model) / (high_model - low_model)
        y_out = np.nan_to_num(woobull_index, nan=0.0)

        ax[0].set_title(self.description)
        ax[0].set_xlabel('Date')
        ax[0].set_ylabel('WoobullIndex')
        sns.lineplot(x=x, y=y_out, ax=ax[0])

        ax[1].set_xlabel('Date')
        ax[1].set_ylabel('Woobull')
        sns.lineplot(x=x, y=woobull, ax=ax[1])
        sns.lineplot(x=x, y=high_model, ax=ax[1])
        sns.lineplot(x=x, y=low_model, ax=ax[1])

        return pl.Series(woobull_index)
