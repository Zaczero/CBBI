import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes

from metrics._common import linreg_predict
from metrics.base_metric import BaseMetric


class TrolololoMetric(BaseMetric):
    @property
    def name(self):
        return 'Trolololo'

    @property
    def description(self):
        return 'Bitcoin Trolololo Trend Line'

    def _calculate(self, df: pl.DataFrame, ax: list[Axes]):
        x = df.get_column('Date').to_numpy()
        price_log = df.get_column('PriceLog').to_numpy()

        begin_date = np.datetime64('2012-01-01T00:00:00')
        days_since_begin = (x - begin_date) / np.timedelta64(1, 'D')

        trolo_top_log = np.log(10.0) * (
            2.900 * np.log(days_since_begin + 1400) - 19.463
        )
        trolo_bottom_log = np.log(10.0) * (
            2.788 * np.log(days_since_begin + 1200) - 19.463
        )

        overshoot_actual = price_log - trolo_top_log
        undershoot_actual = price_log - trolo_bottom_log

        row_nr = np.arange(df.height)
        after_begin = days_since_begin >= 0
        high_mask = df.get_column('PriceHigh').to_numpy() & after_begin
        low_mask = df.get_column('PriceLow').to_numpy() & after_begin

        high_idx = row_nr[high_mask]
        low_idx = row_nr[low_mask]

        high_y = overshoot_actual[high_idx].copy()
        high_y[0] *= 0.6  # the first value seems too high

        overshoot_model = linreg_predict(high_idx, high_y, row_nr)
        undershoot_model = linreg_predict(low_idx, undershoot_actual[low_idx], row_nr)

        high_model = trolo_top_log + overshoot_model
        low_model = trolo_bottom_log + undershoot_model
        trolo_index = (price_log - low_model) / (high_model - low_model)

        y_out = np.nan_to_num(trolo_index, nan=0.0)

        ax[0].set_title(self.description)
        ax[0].set_xlabel('Date')
        ax[0].set_ylabel('TroloIndex')
        sns.lineplot(x=x, y=y_out, ax=ax[0])

        ax[1].set_xlabel('Date')
        ax[1].set_ylabel('PriceLog')
        sns.lineplot(x=x, y=price_log, ax=ax[1])
        sns.lineplot(x=x, y=high_model, ax=ax[1])
        sns.lineplot(x=x, y=low_model, ax=ax[1])

        return pl.Series(trolo_index)
