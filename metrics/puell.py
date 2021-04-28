import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .base_metric import BaseMetric


class PuellMetric(BaseMetric):
    @property
    def name(self) -> str:
        return 'Puell'

    @property
    def description(self) -> str:
        return 'Puell Multiple'

    def calculate(self, source_df: pd.DataFrame) -> pd.Series:
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

        # sns.set()
        # _, ax = plt.subplots()
        # sns.lineplot(x='Date', y='PuellLog', data=df, ax=ax)
        # sns.lineplot(x='Date', y='PuellLogModel', data=df, ax=ax, color='lime')
        # plt.show()

        df['PuellIndex'] = (df['PuellLog'] - projected_min) / \
                           (df['PuellLogModel'] - projected_min)
        return df['PuellIndex']
