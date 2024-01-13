import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from metrics.base_metric import BaseMetric
from utils import HTTP, add_common_markers


def _fetch_df() -> pd.DataFrame:
    response = HTTP.get('https://woocharts.com/bitcoin-price-models/data/chart.json')
    response.raise_for_status()
    data = response.json()

    df_top = pd.DataFrame(
        {
            'Date': data['top_']['x'],
            'Top': data['top_']['y'],
        }
    )
    df_top['Date'] = pd.to_datetime(df_top['Date'], unit='ms').dt.tz_localize(None)

    df_cvdd = pd.DataFrame(
        {
            'Date': data['cvdd']['x'],
            'CVDD': data['cvdd']['y'],
        }
    )
    df_cvdd['Date'] = pd.to_datetime(df_cvdd['Date'], unit='ms').dt.tz_localize(None)

    df = df_top.merge(df_cvdd, on='Date')

    return df


class WoobullMetric(BaseMetric):
    @property
    def name(self) -> str:
        return 'Woobull'

    @property
    def description(self) -> str:
        return 'Woobull Top Cap vs CVDD'

    def _calculate(self, df: pd.DataFrame, ax: list[plt.Axes]) -> pd.Series:
        df = df.merge(_fetch_df(), on='Date', how='left')
        df['Top'].ffill(inplace=True)
        df['TopLog'] = np.log(df['Top'])
        df['CVDD'].ffill(inplace=True)
        df['CVDDLog'] = np.log(df['CVDD'])

        df['Woobull'] = (df['PriceLog'] - df['CVDDLog']) / (df['TopLog'] - df['CVDDLog'])

        high_rows = df.loc[df['PriceHigh'] == 1]
        high_x = high_rows.index.values.reshape(-1, 1)
        high_y = high_rows['Woobull'].values.reshape(-1, 1)

        low_rows = df.loc[df['PriceLow'] == 1][1:]
        low_x = low_rows.index.values.reshape(-1, 1)
        low_y = low_rows['Woobull'].values.reshape(-1, 1)

        x = df.index.values.reshape(-1, 1)

        lin_model = LinearRegression()
        lin_model.fit(high_x, high_y)
        df['WoobullHighModel'] = lin_model.predict(x)

        lin_model.fit(low_x, low_y)
        df['WoobullLowModel'] = lin_model.predict(x)

        df['WoobullIndex'] = (df['Woobull'] - df['WoobullLowModel']) / (df['WoobullHighModel'] - df['WoobullLowModel'])

        ax[0].set_title(self.description)
        sns.lineplot(data=df, x='Date', y='WoobullIndex', ax=ax[0])
        add_common_markers(df, ax[0])

        sns.lineplot(data=df, x='Date', y='Woobull', ax=ax[1])
        sns.lineplot(data=df, x='Date', y='WoobullHighModel', ax=ax[1])
        sns.lineplot(data=df, x='Date', y='WoobullLowModel', ax=ax[1])
        add_common_markers(df, ax[1], price_line=False)

        return df['WoobullIndex']
