import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .base_metric import BaseMetric


class TwoYearMovingAverageMetric(BaseMetric):
    @property
    def name(self) -> str:
        return '2YMA'

    @property
    def description(self) -> str:
        return '2 Year Moving Average'

    def calculate(self, source_df: pd.DataFrame) -> pd.Series:
        df = source_df.copy()

        df['2YMA'] = df['Price'].rolling(365 * 2).mean()
        df['2YMALog'] = np.log(df['2YMA'])
        df['2YMAx5'] = df['2YMA'] * 5
        df['2YMAx5Log'] = np.log(df['2YMAx5'])

        df['2YMALogDifference'] = df['2YMAx5Log'] - df['2YMALog']
        df['2YMALogOvershootActual'] = df['PriceLog'] - df['2YMAx5Log']
        df['2YMALogUndershootActual'] = df['2YMALog'] - df['PriceLog']

        high_rows = df.loc[(df['PriceHigh'] == 1) & ~ (df['2YMA'].isna())]
        high_x = high_rows.index.values.reshape(-1, 1)
        high_y = high_rows['2YMALogOvershootActual'].values.reshape(-1, 1)

        low_rows = df.loc[(df['PriceLow'] == 1) & ~ (df['2YMA'].isna())]
        low_x = low_rows.index.values.reshape(-1, 1)
        low_y = low_rows['2YMALogUndershootActual'].values.reshape(-1, 1)

        x = df.index.values.reshape(-1, 1)

        lin_model = LinearRegression()
        lin_model.fit(high_x, high_y)
        df['2YMALogOvershootModel'] = lin_model.predict(x)

        lin_model.fit(low_x, low_y)
        df['2YMALogUndershootModel'] = lin_model.predict(x)

        # df['2YMALogOvershootModelValue'] = df['2YMAx5Log'] + df['2YMALogOvershootModel']
        # df['2YMALogUndershootModelValue'] = df['2YMALog'] - df['2YMALogUndershootModel']
        # sns.set()
        # _, ax = plt.subplots()
        # sns.lineplot(x='Date', y='PriceLog', data=df, ax=ax)
        # sns.lineplot(x='Date', y='2YMAx5Log', data=df, ax=ax, color='lime')
        # sns.lineplot(x='Date', y='2YMALog', data=df, ax=ax, color='limegreen')
        # sns.lineplot(x='Date', y='2YMALogOvershootModelValue', data=df, ax=ax, color='red')
        # sns.lineplot(x='Date', y='2YMALogUndershootModelValue', data=df, ax=ax, color='orangered')
        # plt.legend(['Price', '2YMA x5', '2YMA', 'Overshoot', 'Undershoot'])
        # plt.show()

        df['2YMAIndex'] = (df['PriceLog'] - df['2YMALog'] + df['2YMALogUndershootModel']) / \
                          (df['2YMALogOvershootModel'] + df['2YMALogDifference'] + df['2YMALogUndershootModel'])
        return df['2YMAIndex']
