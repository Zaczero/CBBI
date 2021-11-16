from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from utils import add_common_markers
from .base_metric import BaseMetric


class HalvingToPeakMetric(BaseMetric):
    @property
    def name(self) -> str:
        return 'HalvingToPeak'

    @property
    def description(self) -> str:
        return 'Halving-to-Peak Days'

    def _calculate(self, df: pd.DataFrame, ax: List[plt.Axes]) -> pd.Series:
        peak_decline_after_days = 30
        peak_decline_duration = 270

        for _, row in df.loc[df['Halving'] == 1].iterrows():
            df.loc[df.index >= row.name, 'PreviousHalvingDate'] = row['Date']

        halving_to_peak_days_list = []

        for _, row in df.loc[df['PriceHigh'] == 1].iterrows():
            halving_to_peak_days = (row['Date'] - row['PreviousHalvingDate']).days
            halving_to_peak_days_list.append([halving_to_peak_days])

            df.loc[((row.name - halving_to_peak_days) <= df.index) & (df.index <= row.name), 'HalvingToPeakDays'] = halving_to_peak_days

        lin_model = LinearRegression()
        lin_model.fit(np.arange(0, len(halving_to_peak_days_list)).reshape(-1, 1), halving_to_peak_days_list)
        halving_to_peak_days_predict = lin_model.predict([[len(halving_to_peak_days_list)]])[0, 0]

        df.loc[(df['DaysSincePriceHigh'] > df['DaysSinceHalving']) & df['HalvingToPeakDays'].isna(), 'HalvingToPeakDays'] = halving_to_peak_days_predict

        df['HalvingToPeakIndex'] = df['DaysSinceHalving'] / df['HalvingToPeakDays']

        df.loc[(peak_decline_after_days >= df['DaysSincePriceHigh']) & (df['DaysSincePriceHigh'] < df['DaysSincePriceLow']), 'HalvingToPeakIndex'] = 1
        df.loc[(peak_decline_after_days < df['DaysSincePriceHigh']) & (df['DaysSincePriceHigh'] < df['DaysSincePriceLow']), 'HalvingToPeakIndex'] = 1 - (df['DaysSincePriceHigh'] - peak_decline_after_days) / peak_decline_duration
        df.loc[(peak_decline_after_days + peak_decline_duration < df['DaysSincePriceHigh']) & (df['DaysSincePriceHigh'] < df['DaysSincePriceLow']), 'HalvingToPeakIndex'] = 0
        df['HalvingToPeakIndex'] = df['HalvingToPeakIndex'].fillna(0)

        ax[0].set_title(self.description)
        sns.lineplot(data=df, x='Date', y='HalvingToPeakIndex', ax=ax[0])
        add_common_markers(df, ax[0])

        return df['HalvingToPeakIndex']
