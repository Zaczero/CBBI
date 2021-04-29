from abc import ABC, abstractmethod
from typing import List

import matplotlib.pyplot as plt
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
    def calculate(self, source_df: pd.DataFrame, ax: List[plt.Axes]) -> pd.Series:
        pass
