import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes

from metrics._common import linreg_predict
from metrics.base_metric import BaseMetric


class TwoYearMovingAverageMetric(BaseMetric):
    @property
    def name(self):
        return '2YMA'

    @property
    def description(self):
        return '2 Year Moving Average'

    def _calculate(self, df: pl.DataFrame, ax: list[Axes]):
        row_nr = np.arange(df.height)
        high_idx = row_nr[df.get_column('PriceHigh').to_numpy()]
        low_idx = row_nr[df.get_column('PriceLow').to_numpy()]

        price_log = df.get_column('PriceLog').to_numpy()
        two_yma = df.get_column('Price730DMA').to_numpy()
        two_yma_log = np.log(two_yma)
        log_diff = price_log - two_yma_log

        overshoot_model = linreg_predict(high_idx, log_diff[high_idx], row_nr)
        undershoot_model = linreg_predict(low_idx, log_diff[low_idx], row_nr)

        x = df.get_column('Date').to_numpy()
        two_yma_high_model = overshoot_model + two_yma_log
        two_yma_low_model = undershoot_model + two_yma_log
        two_yma_index = (price_log - two_yma_low_model) / (
            two_yma_high_model - two_yma_low_model
        )
        y_out = np.nan_to_num(two_yma_index, nan=0.0)

        ax[0].set_title(self.description)
        ax[0].set_xlabel('Date')
        ax[0].set_ylabel('2YMAIndex')
        sns.lineplot(x=x, y=y_out, ax=ax[0])

        ax[1].set_xlabel('Date')
        ax[1].set_ylabel('PriceLog')
        sns.lineplot(x=x, y=price_log, ax=ax[1])
        sns.lineplot(x=x, y=two_yma_high_model, ax=ax[1])
        sns.lineplot(x=x, y=two_yma_low_model, ax=ax[1])

        return pl.Series(two_yma_index)
