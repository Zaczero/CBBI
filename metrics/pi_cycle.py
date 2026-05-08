import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes

from metrics._common import linreg_predict
from metrics.base_metric import BaseMetric
from utils import mark_highs_lows


class PiCycleMetric(BaseMetric):
    @property
    def name(self):
        return 'PiCycle'

    @property
    def description(self):
        return 'Pi Cycle Top Indicator'

    def _calculate(self, df: pl.DataFrame, ax: list[Axes]):
        dma_111 = pl.col('Price').rolling_mean(window_size=111, min_samples=111)
        dma_350x2 = pl.col('Price').rolling_mean(window_size=350, min_samples=350) * 2

        df = df.with_columns(
            dma_111.alias('111DMA'),
            dma_350x2.alias('350DMAx2'),
        ).with_columns(PiCycleDiff=pl.col('111DMA').log() - pl.col('350DMAx2').log())

        diff = df.get_column('PiCycleDiff').to_numpy()
        df = mark_highs_lows(df, 'PiCycleDiff', True, 365 * 2, 365)

        crossed = diff > 0
        crossed_idx = np.flatnonzero(crossed)
        uncrossed_idx = np.flatnonzero(~crossed)
        crossed_segments = (
            np.split(crossed_idx, np.where(np.diff(crossed_idx) > 1)[0] + 1)
            if crossed_idx.size
            else []
        )
        uncrossed_segments = (
            np.split(uncrossed_idx, np.where(np.diff(uncrossed_idx) > 1)[0] + 1)
            if uncrossed_idx.size
            else []
        )

        distance_floor = np.zeros(df.height, dtype=np.float64)
        distance_from_zero = np.abs(diff)
        for i, crossed_seg in enumerate(crossed_segments):
            seg_distance = distance_from_zero[crossed_seg]
            max_pos = int(np.nanargmax(seg_distance))
            max_idx = int(crossed_seg[max_pos])
            max_distance = float(seg_distance[max_pos])

            distance_floor[max_idx + 1 :] = max_distance

            uncrossed_seg = (
                uncrossed_segments[i + 1] if i + 1 < len(uncrossed_segments) else None
            )
            if uncrossed_seg is not None and uncrossed_seg.size:
                above = uncrossed_seg[distance_from_zero[uncrossed_seg] >= max_distance]
                if above.size:
                    distance_floor[int(above.min()) :] = 0

            crossed_next = (
                crossed_segments[i + 1] if i + 1 < len(crossed_segments) else None
            )
            if crossed_next is not None and crossed_next.size:
                distance_floor[int(crossed_next.min()) :] = 0

        high_idx = np.flatnonzero(df.get_column('PiCycleDiffHigh').to_numpy())
        row_nr = np.arange(df.height)
        low_idx = np.flatnonzero(df.get_column('PiCycleDiffLow').to_numpy())

        target = np.zeros(df.height, dtype=np.float64)
        if high_idx.size >= 3:
            target = np.minimum(
                linreg_predict(high_idx, diff[high_idx], row_nr),
                0.0,
            )

        finite_idx = np.flatnonzero(np.isfinite(diff))
        cold_model = np.full(df.height, np.nan, dtype=np.float64)

        cycle_starts = np.array([int(finite_idx[0]), *low_idx], dtype=np.int64)
        for i, start in enumerate(cycle_starts):
            end = (
                int(cycle_starts[i + 1] - 1)
                if i + 1 < cycle_starts.size
                else df.height - 1
            )
            cold_model[start : end + 1] = diff[start]

        distance = np.maximum(np.maximum(target - diff, 0.0), distance_floor)
        index = 1 - distance / np.abs(target - cold_model)

        x = df.get_column('Date').to_numpy()
        y_out = np.nan_to_num(index, nan=0.0)

        ax[0].set_title(self.description)
        ax[0].set_xlabel('Date')
        ax[0].set_ylabel('PiCycleIndex')
        sns.lineplot(x=x, y=y_out, ax=ax[0])

        ax[1].set_xlabel('Date')
        ax[1].set_ylabel('PiCycleDiff')
        sns.lineplot(x=x, y=diff, ax=ax[1])
        sns.lineplot(x=x, y=target, ax=ax[1])
        sns.lineplot(x=x, y=cold_model, ax=ax[1])
        sns.lineplot(x=x, y=target - distance_floor, ax=ax[1], linestyle='--')

        return pl.Series('PiCycleIndex', index)
