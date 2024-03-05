import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from api.coinsoto_api import cs_fetch
from metrics.base_metric import BaseMetric
from utils import add_common_markers


class ReserveRiskMetric(BaseMetric):
    @property
    def name(self) -> str:
        return 'ReserveRisk'

    @property
    def description(self) -> str:
        return 'Reserve Risk'

    def _calculate(self, df: pd.DataFrame, ax: list[plt.Axes]) -> pd.Series:
        days_shift = 1

        df = df.merge(
            cs_fetch(
                path='chain/index/charts?type=/charts/reserve-risk/',
                data_selector='value4',
                col_name='Risk',
            ),
            on='Date',
            how='left',
        )
        df['Risk'] = df['Risk'].shift(days_shift, fill_value=np.nan)
        df['Risk'] = df['Risk'].ffill()
        df['RiskLog'] = np.log(df['Risk'])

        high_rows = df.loc[df['PriceHigh'] == 1]
        high_x = high_rows.index.values.reshape(-1, 1)
        high_y = high_rows['RiskLog'].values.reshape(-1, 1)

        low_rows = df.loc[df['PriceLow'] == 1][1:]
        low_x = low_rows.index.values.reshape(-1, 1)
        low_y = low_rows['RiskLog'].values.reshape(-1, 1)

        x = df.index.values.reshape(-1, 1)

        lin_model = LinearRegression()
        lin_model.fit(high_x, high_y)
        df['HighModel'] = lin_model.predict(x)
        df['HighModel'] = df['HighModel'] - 0.15

        lin_model.fit(low_x, low_y)
        df['LowModel'] = lin_model.predict(x)

        df['RiskIndex'] = (df['RiskLog'] - df['LowModel']) / (df['HighModel'] - df['LowModel'])

        df['RiskIndexNoNa'] = df['RiskIndex'].fillna(0)
        ax[0].set_title(self.description)
        sns.lineplot(data=df, x='Date', y='RiskIndexNoNa', ax=ax[0])
        add_common_markers(df, ax[0])

        sns.lineplot(data=df, x='Date', y='RiskLog', ax=ax[1])
        sns.lineplot(data=df, x='Date', y='HighModel', ax=ax[1])
        sns.lineplot(data=df, x='Date', y='LowModel', ax=ax[1])
        add_common_markers(df, ax[1], price_line=False)

        return df['RiskIndex']
