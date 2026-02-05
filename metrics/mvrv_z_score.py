import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes

from api.coinsoto_api import cs_fetch
from metrics._common import join_left_on_date, linreg_predict
from metrics.base_metric import BaseMetric


class MVRVMetric(BaseMetric):
    @property
    def name(self):
        return 'MVRV'

    @property
    def description(self):
        return 'MVRV Z-Score'

    def _calculate(self, df: pl.DataFrame, ax: list[Axes]):
        bull_days_shift = 6
        low_model_adjust = 0.26

        df = join_left_on_date(
            df,
            cs_fetch(
                path='chain/index/charts?type=/charts/mvrv-zscore/',
                data_selector='value4',
                col_name='MVRV',
            ),
        )

        df = df.with_columns(
            MVRV=(
                pl
                .when(pl.col('DaysSinceHalving') < pl.col('DaysSincePriceLow'))
                .then(pl.col('MVRV').shift(bull_days_shift))
                .otherwise(pl.col('MVRV'))
            )
        )
        df = df.with_columns(MVRV=(pl.col('MVRV').forward_fill() + 1).log())

        row_nr = np.arange(df.height)
        high_idx = row_nr[df.get_column('PriceHigh').to_numpy()]
        low_idx = row_nr[df.get_column('PriceLow').to_numpy()]

        mvrv = df.get_column('MVRV').to_numpy()
        high_model = linreg_predict(high_idx, mvrv[high_idx], row_nr)
        low_model = linreg_predict(low_idx, mvrv[low_idx], row_nr) + low_model_adjust

        x = df.get_column('Date').to_numpy()
        mvrv_index = (mvrv - low_model) / (high_model - low_model)
        y_out = np.nan_to_num(mvrv_index, nan=0.0)

        ax[0].set_title(self.description)
        ax[0].set_xlabel('Date')
        ax[0].set_ylabel('MVRVIndex')
        sns.lineplot(x=x, y=y_out, ax=ax[0])

        ax[1].set_xlabel('Date')
        ax[1].set_ylabel('MVRV')
        sns.lineplot(x=x, y=mvrv, ax=ax[1])
        sns.lineplot(x=x, y=high_model, ax=ax[1])
        sns.lineplot(x=x, y=low_model, ax=ax[1])

        return pl.Series(mvrv_index)
