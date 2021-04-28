from abc import ABC, abstractmethod

import pandas as pd


class BaseMetric(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def calculate(self, source_df: pd.DataFrame) -> pd.Series:
        pass
