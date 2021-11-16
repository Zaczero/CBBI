import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from api.lookintobitcoin_api import lib_fetch
from metrics.base_metric import BaseMetric
from utils import add_common_markers


class PuellMetric(BaseMetric):
    @property
    def name(self) -> str:
        return 'Puell'

    @property
    def description(self) -> str:
        return 'Puell Multiple'

    def _calculate(self, df: pd.DataFrame, ax: list[plt.Axes]) -> pd.Series:
        df = df.merge(lib_fetch(
            url_selector='puell_multiple',
            post_selector='puell_multiple',
            chart_idx=1,
            col_name='Puell'
        ), on='Date', how='left')
        df['Puell'].ffill(inplace=True)
        df['PuellLog'] = np.log(df['Puell'])

        high_rows = df.loc[df['PriceHigh'] == 1]
        high_x = high_rows.index.values.reshape(-1, 1)
        high_y = high_rows['PuellLog'].values.reshape(-1, 1)

        low_rows = df.loc[df['PriceLow'] == 1][1:]
        low_x = low_rows.index.values.reshape(-1, 1)
        low_y = low_rows['PuellLog'].values.reshape(-1, 1)

        x = df.index.values.reshape(-1, 1)

        lin_model = LinearRegression()
        lin_model.fit(high_x, high_y)
        df['PuellLogHighModel'] = lin_model.predict(x)

        lin_model.fit(low_x, low_y)
        df['PuellLogLowModel'] = lin_model.predict(x)

        df['PuellIndex'] = (df['PuellLog'] - df['PuellLogLowModel']) / \
                           (df['PuellLogHighModel'] - df['PuellLogLowModel'])

        df['PuellIndexNoNa'] = df['PuellIndex'].fillna(0)
        ax[0].set_title(self.description)
        sns.lineplot(data=df, x='Date', y='PuellIndexNoNa', ax=ax[0])
        add_common_markers(df, ax[0])

        sns.lineplot(data=df, x='Date', y='PuellLog', ax=ax[1])
        sns.lineplot(data=df, x='Date', y='PuellLogHighModel', ax=ax[1])
        sns.lineplot(data=df, x='Date', y='PuellLogLowModel', ax=ax[1])
        add_common_markers(df, ax[1], price_line=False)

        return df['PuellIndex']
