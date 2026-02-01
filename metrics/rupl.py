import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from sklearn.linear_model import LinearRegression

from api.bgeometrics_api import bg_fetch
from metrics.base_metric import BaseMetric
from utils import add_common_markers


class RUPLMetric(BaseMetric):
    @property
    def name(self) -> str:
        return 'RUPL'

    @property
    def description(self) -> str:
        return 'RUPL/NUPL Chart'

    def _calculate(self, df: pd.DataFrame, ax: list[Axes]) -> pd.Series:
        df = df.merge(
            bg_fetch(
                endpoint='nupl',
                value_col='nupl',
                col_name='RUPL',
            ),
            on='Date',
            how='left',
        )
        df['RUPL'] = df['RUPL'].ffill()

        high_rows = df.loc[df['PriceHigh'] == 1]
        high_x = high_rows.index.values.reshape(-1, 1)
        high_y = high_rows['RUPL'].values.reshape(-1, 1)

        low_rows = df.loc[df['PriceLow'] == 1][1:]
        low_x = low_rows.index.values.reshape(-1, 1)
        low_y = low_rows['RUPL'].values.reshape(-1, 1)

        x = df.index.values.reshape(-1, 1)

        lin_model = LinearRegression()
        lin_model.fit(high_x, high_y)
        df['HighModel'] = lin_model.predict(x)

        lin_model.fit(low_x, low_y)
        df['LowModel'] = lin_model.predict(x)

        df['RUPLIndex'] = (df['RUPL'] - df['LowModel']) / (df['HighModel'] - df['LowModel'])

        ax[0].set_title(self.description)
        sns.lineplot(data=df, x='Date', y='RUPLIndex', ax=ax[0])
        add_common_markers(df, ax[0])

        sns.lineplot(data=df, x='Date', y='RUPL', ax=ax[1])
        sns.lineplot(data=df, x='Date', y='HighModel', ax=ax[1])
        sns.lineplot(data=df, x='Date', y='LowModel', ax=ax[1])
        add_common_markers(df, ax[1], price_line=False)

        return df['RUPLIndex']
