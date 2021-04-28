from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from utils import add_common_markers
from .base_metric import BaseMetric


class PuellMetric(BaseMetric):
    @property
    def name(self) -> str:
        return 'Puell'

    @property
    def description(self) -> str:
        return 'Puell Multiple'

    def calculate(self, source_df: pd.DataFrame, ax: List[plt.Axes]) -> pd.Series:
        projected_min = np.log(0.3)

        df = source_df.copy()

        df['PuellMA365'] = df['TotalGenerationUSD'].rolling(365).mean()
        df['Puell'] = df['TotalGenerationUSD'] / df['PuellMA365']
        df['PuellLog'] = np.log(df['Puell'])

        high_rows = df.loc[(df['PriceHigh'] == 1) & ~ (df['Puell'].isna())]
        high_x = high_rows.index.values.reshape(-1, 1)
        high_y = high_rows['PuellLog'].values.reshape(-1, 1)

        x = df.index.values.reshape(-1, 1)

        lin_model = LinearRegression()
        lin_model.fit(high_x, high_y)
        df['PuellLogModel'] = lin_model.predict(x)

        df['PuellIndex'] = (df['PuellLog'] - projected_min) / \
                           (df['PuellLogModel'] - projected_min)

        df['PuellIndexNoNa'] = df['PuellIndex'].fillna(0)
        ax[0].set_title(self.description)
        sns.lineplot(data=df, x='Date', y='PuellIndexNoNa', ax=ax[0])
        add_common_markers(df, ax[0])

        return df['PuellIndex']
