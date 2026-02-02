import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from sklearn.linear_model import LinearRegression

from metrics.base_metric import BaseMetric
from utils import add_common_markers, mark_highs_lows


class PuellMetric(BaseMetric):
    @property
    def name(self) -> str:
        return 'Puell'

    @property
    def description(self) -> str:
        return 'Puell Multiple'

    def _calculate(self, df: pd.DataFrame, ax: list[Axes]) -> pd.Series:
        # Calculate Puell Multiple locally from mining revenue data
        # Puell = daily_mining_revenue / 365-day_MA_of_mining_revenue
        # TotalGenerationUSD contains daily mining revenue in USD from Blockchain.com
        df['Puell'] = df['TotalGenerationUSD'] / df['TotalGenerationUSD'].rolling(window=365, min_periods=1).mean()
        df['Puell'] = df['Puell'].ffill()
        df['PuellLog'] = np.log(df['Puell'])

        df = mark_highs_lows(df, 'PuellLog', True, round(365 * 2), 365)
        high_rows = df.loc[(df['PuellLogHigh'] == 1) & (df.index > 365)]
        
        high_x = high_rows.index.values.reshape(-1, 1)
        high_y = high_rows['PuellLog'].values.reshape(-1, 1)

        # low_rows = df.loc[df['PriceLow'] == 1][1:]
        # low_x = low_rows.index.values.reshape(-1, 1)
        # low_y = low_rows['PuellLog'].values.reshape(-1, 1)

        x = df.index.values.reshape(-1, 1)

        lin_model = LinearRegression()
        lin_model.fit(high_x, high_y)
        predictions = lin_model.predict(x)
        min_peak = high_y.min()
        df['PuellLogHighModel'] = np.maximum(predictions, min_peak)

        # lin_model.fit(low_x, low_y)
        # df['PuellLogLowModel'] = lin_model.predict(x)
        df['PuellLogLowModel'] = -1

        df['PuellIndex'] = (df['PuellLog'] - df['PuellLogLowModel']) / (
            df['PuellLogHighModel'] - df['PuellLogLowModel']
        )

        df['PuellIndexNoNa'] = df['PuellIndex'].fillna(0)
        ax[0].set_title(self.description)
        sns.lineplot(data=df, x='Date', y='PuellIndexNoNa', ax=ax[0])
        add_common_markers(df, ax[0])

        sns.lineplot(data=df, x='Date', y='PuellLog', ax=ax[1])
        sns.lineplot(data=df, x='Date', y='PuellLogHighModel', ax=ax[1])
        sns.lineplot(data=df, x='Date', y='PuellLogLowModel', ax=ax[1])
        add_common_markers(df, ax[1], price_line=False)

        return df['PuellIndex']
