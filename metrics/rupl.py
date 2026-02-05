import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes

from api.coinsoto_api import cs_fetch
from metrics._common import join_left_on_date, linreg_predict
from metrics.base_metric import BaseMetric


class RUPLMetric(BaseMetric):
    @property
    def name(self):
        return 'RUPL'

    @property
    def description(self):
        return 'RUPL/NUPL Chart'

    def _calculate(self, df: pl.DataFrame, ax: list[Axes]):
        df = join_left_on_date(
            df,
            cs_fetch(
                path='chain/index/charts?type=/charts/relative-unrealized-prof/',
                data_selector='value1',
                col_name='RUPL',
            ),
        )

        df = df.with_columns(RUPL=pl.col('RUPL').forward_fill())

        row_nr = np.arange(df.height)
        high_idx = row_nr[df.get_column('PriceHigh').to_numpy()]
        low_idx = row_nr[df.get_column('PriceLow').to_numpy()][1:]

        rupl = df.get_column('RUPL').to_numpy()
        x_all = row_nr

        high_model = linreg_predict(high_idx, rupl[high_idx], x_all)
        low_model = linreg_predict(low_idx, rupl[low_idx], x_all)

        x = df.get_column('Date').to_numpy()
        rupl_index = (rupl - low_model) / (high_model - low_model)
        y_out = np.nan_to_num(rupl_index, nan=0.0)

        ax[0].set_title(self.description)
        ax[0].set_xlabel('Date')
        ax[0].set_ylabel('RUPLIndex')
        sns.lineplot(x=x, y=y_out, ax=ax[0])

        ax[1].set_xlabel('Date')
        ax[1].set_ylabel('RUPL')
        sns.lineplot(x=x, y=rupl, ax=ax[1])
        sns.lineplot(x=x, y=high_model, ax=ax[1])
        sns.lineplot(x=x, y=low_model, ax=ax[1])

        return pl.Series(rupl_index)
