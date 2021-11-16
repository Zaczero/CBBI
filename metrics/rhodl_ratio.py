import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from api.lookintobitcoin_api import lib_fetch
from utils import add_common_markers
from metrics.base_metric import BaseMetric


class RHODLMetric(BaseMetric):
    @property
    def name(self) -> str:
        return 'RHODL'

    @property
    def description(self) -> str:
        return 'RHODL Ratio'

    def _calculate(self, df: pd.DataFrame, ax: list[plt.Axes]) -> pd.Series:
        df = df.merge(lib_fetch(
            url_selector='rhodl_ratio',
            post_selector='rhodl-ratio',
            chart_idx=1,
            col_name='RHODL'
        ), on='Date', how='left')
        df['RHODL'].ffill(inplace=True)
        df['RHODLLog'] = np.log(df['RHODL'])

        high_rows = df.loc[df['PriceHigh'] == 1]
        high_x = high_rows.index.values.reshape(-1, 1)
        high_y = high_rows['RHODLLog'].values.reshape(-1, 1)

        low_rows = df.loc[df['PriceLow'] == 1][1:]
        low_x = low_rows.index.values.reshape(-1, 1)
        low_y = low_rows['RHODLLog'].values.reshape(-1, 1)

        x = df.index.values.reshape(-1, 1)

        lin_model = LinearRegression()
        lin_model.fit(high_x, high_y)
        df['RHODLLogHighModel'] = lin_model.predict(x)

        lin_model.fit(low_x, low_y)
        df['RHODLLogLowModel'] = lin_model.predict(x)

        df['RHODLIndex'] = (df['RHODLLog'] - df['RHODLLogLowModel']) / \
                           (df['RHODLLogHighModel'] - df['RHODLLogLowModel'])

        df['RHODLIndexNoNa'] = df['RHODLIndex'].fillna(0)
        ax[0].set_title(self.description)
        sns.lineplot(data=df, x='Date', y='RHODLIndexNoNa', ax=ax[0])
        add_common_markers(df, ax[0])

        sns.lineplot(data=df, x='Date', y='RHODLLog', ax=ax[1])
        sns.lineplot(data=df, x='Date', y='RHODLLogHighModel', ax=ax[1])
        sns.lineplot(data=df, x='Date', y='RHODLLogLowModel', ax=ax[1])
        add_common_markers(df, ax[1], price_line=False)

        return df['RHODLIndex']
