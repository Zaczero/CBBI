import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes

from api.coinsoto_api import cs_fetch
from metrics._common import join_left_on_date, linreg_predict
from metrics.base_metric import BaseMetric


class RHODLMetric(BaseMetric):
    @property
    def name(self):
        return 'RHODL'

    @property
    def description(self):
        return 'RHODL Ratio'

    def _calculate(self, df: pl.DataFrame, ax: list[Axes]):
        remote_df = cs_fetch(
            path='chain/index/charts?type=/charts/rhodl-ratio/',
            data_selector='value1',
            col_name='RHODL',
        )

        df = join_left_on_date(df, remote_df).with_columns(
            RHODL=pl.col('RHODL').forward_fill()
        )

        row_nr = np.arange(df.height)
        high_mask = df.get_column('PriceHigh').to_numpy() | (
            df.get_column('Date').to_numpy() == np.datetime64('2024-12-18')
        )
        high_idx = row_nr[high_mask]
        low_idx = row_nr[df.get_column('PriceLow').to_numpy()][1:]

        rhodl = df.get_column('RHODL').to_numpy()
        rhodl_log = np.log(rhodl)

        high_model = linreg_predict(high_idx, rhodl_log[high_idx], row_nr)
        low_model = linreg_predict(low_idx, rhodl_log[low_idx], row_nr)

        x = df.get_column('Date').to_numpy()
        rhodl_index = (rhodl_log - low_model) / (high_model - low_model)
        y_out = np.nan_to_num(rhodl_index, nan=0.0)

        ax[0].set_title(self.description)
        ax[0].set_xlabel('Date')
        ax[0].set_ylabel('RHODLIndex')
        sns.lineplot(x=x, y=y_out, ax=ax[0])

        ax[1].set_xlabel('Date')
        ax[1].set_ylabel('RHODLLog')
        sns.lineplot(x=x, y=rhodl_log, ax=ax[1])
        sns.lineplot(x=x, y=high_model, ax=ax[1])
        sns.lineplot(x=x, y=low_model, ax=ax[1])

        return pl.Series(rhodl_index)
