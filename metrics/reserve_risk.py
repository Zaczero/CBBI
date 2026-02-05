import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes

from api.coinsoto_api import cs_fetch
from metrics._common import join_left_on_date, linreg_predict
from metrics.base_metric import BaseMetric


class ReserveRiskMetric(BaseMetric):
    @property
    def name(self):
        return 'ReserveRisk'

    @property
    def description(self):
        return 'Reserve Risk'

    def _calculate(self, df: pl.DataFrame, ax: list[Axes]):
        days_shift = 1

        df = join_left_on_date(
            df,
            cs_fetch(
                path='chain/index/charts?type=/charts/reserve-risk/',
                data_selector='value4',
                col_name='Risk',
            ),
        )
        df = df.with_columns(Risk=pl.col('Risk').shift(days_shift).forward_fill())

        row_nr = np.arange(df.height)
        high_idx = row_nr[df.get_column('PriceHigh').to_numpy()]
        low_idx = row_nr[df.get_column('PriceLow').to_numpy()][1:]

        risk_log = np.log(df.get_column('Risk').to_numpy())
        high_model = linreg_predict(high_idx, risk_log[high_idx], row_nr) - 0.15
        low_model = linreg_predict(low_idx, risk_log[low_idx], row_nr)

        x = df.get_column('Date').to_numpy()
        risk_index = (risk_log - low_model) / (high_model - low_model)
        y_out = np.nan_to_num(risk_index, nan=0.0)

        ax[0].set_title(self.description)
        ax[0].set_xlabel('Date')
        ax[0].set_ylabel('RiskIndex')
        sns.lineplot(x=x, y=y_out, ax=ax[0])

        ax[1].set_xlabel('Date')
        ax[1].set_ylabel('RiskLog')
        sns.lineplot(x=x, y=risk_log, ax=ax[1])
        sns.lineplot(x=x, y=high_model, ax=ax[1])
        sns.lineplot(x=x, y=low_model, ax=ax[1])

        return pl.Series(risk_index)
