import math
import traceback
from abc import ABC, abstractmethod

import polars as pl
from matplotlib.axes import Axes
from sty import bg, fg, rs

from api.cbbiinfo_api import cbbi_fetch
from utils import send_error_notification


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
    def _calculate(self, df: pl.DataFrame, ax: list[Axes]) -> pl.Series:
        pass

    def _fallback(self, df: pl.DataFrame):
        return (
            df
            .join(cbbi_fetch(self.name), on='Date', how='left', maintain_order='left')
            .with_columns(pl.col('Value').forward_fill())
            .get_column('Value')
        )

    def _require_current_value(self, values: pl.Series):
        current = values[-1] if len(values) else None
        if current is None or not math.isfinite(current):
            raise ValueError(f'{self.description} has no finite current value')
        return values

    async def calculate(self, df: pl.DataFrame, ax: list[Axes]):
        try:
            return self._require_current_value(self._calculate(df, ax))
        except Exception as ex:
            traceback.print_exc()
            try:
                await send_error_notification(ex)
            except Exception:
                traceback.print_exc()

            print(
                fg.black
                + bg.yellow
                + f' Requesting fallback values for {self.name} (from CBBI.info) '
                + rs.all
            )
            return self._require_current_value(self._fallback(df))
