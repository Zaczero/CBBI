import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from api.lookintobitcoin_api import lib_fetch
from utils import mark_highs_lows, add_common_markers
from metrics.base_metric import BaseMetric


class RUPLMetric(BaseMetric):
    @property
    def name(self) -> str:
        return 'RUPL'

    @property
    def description(self) -> str:
        return 'RUPL/NUPL Chart'

    def _calculate(self, df: pd.DataFrame, ax: list[plt.Axes]) -> pd.Series:
        df = df.merge(lib_fetch(
            url_selector='unrealised_profit_loss',
            post_selector='relative-unrealized-profit--loss',
            chart_idx='Net Unrealised Profit / Loss (NUPL)',
            col_name='RUPL'
        ), on='Date', how='left')
        df['RUPL'].ffill(inplace=True)

        df = mark_highs_lows(df, 'RUPL', False, 120, 120)
        df.fillna({'RUPLHigh': 0, 'RUPLLow': 0}, inplace=True)

        df.loc[df['RUPL'] < 0.75, 'RUPLHigh'] = 0

        high_rows = df.loc[df['RUPLHigh'] == 1]
        high_x = high_rows.index.values.reshape(-1, 1)
        high_y = high_rows['RUPL'].values.reshape(-1, 1)

        low_rows = df.loc[df['PriceLow'] == 1][1:]
        low_x = low_rows.index.values.reshape(-1, 1)
        low_y = low_rows['RUPL'].values.reshape(-1, 1)

        x = df.index.values.reshape(-1, 1)

        lin_model = LinearRegression()
        lin_model.fit(high_x, high_y)
        df['RUPLHighModel'] = lin_model.predict(x)

        lin_model.fit(low_x, low_y)
        df['RUPLLowModel'] = lin_model.predict(x)

        df['RUPLIndex'] = (df['RUPL'] - df['RUPLLowModel']) / \
                          (df['RUPLHighModel'] - df['RUPLLowModel'])

        ax[0].set_title(self.description)
        sns.lineplot(data=df, x='Date', y='RUPLIndex', ax=ax[0])
        add_common_markers(df, ax[0])

        sns.lineplot(data=df, x='Date', y='RUPL', ax=ax[1])
        sns.lineplot(data=df, x='Date', y='RUPLHighModel', ax=ax[1])
        sns.lineplot(data=df, x='Date', y='RUPLLowModel', ax=ax[1])
        add_common_markers(df, ax[1], price_line=False)

        return df['RUPLIndex']
