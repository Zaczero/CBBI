from datetime import timedelta

import numpy as np
import pandas as pd

from .base_metric import BaseMetric


class StockToFlowMetric(BaseMetric):
    @property
    def name(self) -> str:
        return 'StockToFlow'

    @property
    def description(self) -> str:
        return 'Stock-to-Flow Chart'

    def calculate(self, source_df: pd.DataFrame) -> pd.Series:
        sf_emerge_days = 463
        sf_peak_delay = 60

        df = source_df.copy()

        for _, row in df.loc[df['PriceHigh'] == 1].iterrows():
            df.loc[df.index > row.name, 'PreviousPriceHighDate'] = row['Date']

        for _, row in df.loc[df['Halving'] == 1].iterrows():
            df.loc[df.index > row.name, 'PreviousHalvingDate'] = row['Date']

        df['StockToFlowTarget'] = df['PreviousHalvingDate'] + timedelta(sf_emerge_days + sf_peak_delay)
        df['StockToFlow'] = (df['Date'] - df['PreviousPriceHighDate']) / \
                            (df['StockToFlowTarget'] - df['PreviousPriceHighDate'])

        df['StockToFlowIndex'] = 1 - np.abs(1 - df['StockToFlow'])
        df.loc[df['PreviousPriceHighDate'] >= df['PreviousHalvingDate'], 'StockToFlowIndex'] = np.nan
        return df['StockToFlowIndex']
