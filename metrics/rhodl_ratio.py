import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from sklearn.linear_model import LinearRegression
from sty import bg, fg, rs

from api.bgeometrics_api import bg_fetch
from api.glassnode_api import gn_fetch
from metrics.base_metric import BaseMetric
from utils import add_common_markers


class RHODLMetric(BaseMetric):
    @property
    def name(self) -> str:
        return 'RHODL'

    @property
    def description(self) -> str:
        return 'RHODL Ratio'

    def _calculate(self, df: pd.DataFrame, ax: list[Axes]) -> pd.Series:
        try:
            remote_df = bg_fetch(
                endpoint='rhodl-ratio',
                value_col='rhodlRatio',
                col_name='RHODL',
            )
        except Exception:
            # Print full traceback for backend debugging
            import traceback
            traceback.print_exc()
            print(fg.black + bg.yellow + f' Requesting fallback values for {self.name} (from GlassNode) ' + rs.all)

            remote_df = gn_fetch(url_selector='rhodl_ratio', col_name='RHODL', a='BTC')

        df = df.merge(remote_df, on='Date', how='left')
        df['RHODL'] = df['RHODL'].ffill()
        df['RHODLLog'] = np.log(df['RHODL'])

        high_rows = df.loc[(df['PriceHigh'] == 1) | (df['Date'] == '2024-12-18')]
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

        df['RHODLIndex'] = (df['RHODLLog'] - df['RHODLLogLowModel']) / (
            df['RHODLLogHighModel'] - df['RHODLLogLowModel']
        )

        df['RHODLIndexNoNa'] = df['RHODLIndex'].fillna(0)
        ax[0].set_title(self.description)
        sns.lineplot(data=df, x='Date', y='RHODLIndexNoNa', ax=ax[0])
        add_common_markers(df, ax[0])

        sns.lineplot(data=df, x='Date', y='RHODLLog', ax=ax[1])
        sns.lineplot(data=df, x='Date', y='RHODLLogHighModel', ax=ax[1])
        sns.lineplot(data=df, x='Date', y='RHODLLogLowModel', ax=ax[1])
        add_common_markers(df, ax[1], price_line=False)

        return df['RHODLIndex']
