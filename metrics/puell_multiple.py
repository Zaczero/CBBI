import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes

from metrics._common import linreg_predict
from metrics.base_metric import BaseMetric


class PuellMetric(BaseMetric):
    @property
    def name(self):
        return 'Puell'

    @property
    def description(self):
        return 'Puell Multiple'

    def _calculate(self, df: pl.DataFrame, ax: list[Axes]):
        puell_log = (
            df
            .get_column('PuellMultiple')
            .forward_fill()
            .log()
            .rolling_mean(window_size=3, min_samples=1)
            .to_numpy()
        )

        row_nr = np.arange(df.height)
        high_idx = row_nr[df.get_column('PriceHigh').to_numpy()]

        high_model = linreg_predict(high_idx, puell_log[high_idx], row_nr)
        low_model = -1.0

        x = df.get_column('Date').to_numpy()
        puell_index = (puell_log - low_model) / (high_model - low_model)
        y_out = np.nan_to_num(puell_index, nan=0.0)

        ax[0].set_title(self.description)
        ax[0].set_xlabel('Date')
        ax[0].set_ylabel('PuellIndex')
        sns.lineplot(x=x, y=y_out, ax=ax[0])

        ax[1].set_xlabel('Date')
        ax[1].set_ylabel('PuellLog')
        sns.lineplot(x=x, y=puell_log, ax=ax[1])
        sns.lineplot(x=x, y=high_model, ax=ax[1])
        sns.lineplot(x=x, y=np.full(df.height, low_model, dtype=np.float64), ax=ax[1])

        return pl.Series(puell_index)
