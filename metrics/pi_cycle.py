import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes

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
        ).with_columns(
            PiCycleDiff=(pl.col('111DMA').log() - pl.col('350DMAx2').log()).abs()
        )

        diff = df.get_column('PiCycleDiff').to_numpy()
        dma_111 = df.get_column('111DMA').to_numpy()
        dma_350x2 = df.get_column('350DMAx2').to_numpy()
        fluke = (dma_111 > dma_350x2) & ~np.isnan(dma_111) & ~np.isnan(dma_350x2)

        idx_fluke = np.flatnonzero(fluke)
        idx_actual = np.flatnonzero(~fluke)
        fluke_segments = (
            np.split(idx_fluke, np.where(np.diff(idx_fluke) > 1)[0] + 1)
            if idx_fluke.size
            else []
        )
        actual_segments = (
            np.split(idx_actual, np.where(np.diff(idx_actual) > 1)[0] + 1)
            if idx_actual.size
            else []
        )

        threshold = np.zeros(len(diff), dtype=np.float64)
        for i, fluke_seg in enumerate(fluke_segments):
            if fluke_seg.size == 0:
                continue

            seg_diff = diff[fluke_seg]
            max_pos = int(np.nanargmax(seg_diff))
            max_idx = int(fluke_seg[max_pos])
            max_val = float(seg_diff[max_pos])

            threshold[max_idx + 1 :] = max_val

            actual_seg = (
                actual_segments[i + 1] if i + 1 < len(actual_segments) else None
            )
            if actual_seg is not None and actual_seg.size:
                actual_diff = diff[actual_seg]
                above = actual_seg[actual_diff >= max_val]
                if above.size:
                    threshold[int(above.min()) :] = 0

            fluke_next = fluke_segments[i + 1] if i + 1 < len(fluke_segments) else None
            if fluke_next is not None and fluke_next.size:
                threshold[int(fluke_next.min()) :] = 0

        diff = np.where(diff < threshold, threshold, diff)

        df = df.with_columns(
            pl.Series('PiCycleDiffThreshold', threshold),
            pl.Series('PiCycleDiff', diff),
        )
        df = mark_highs_lows(df, 'PiCycleDiff', True, 365 * 2, 365)

        high_marks = df.get_column('PiCycleDiffHigh').to_numpy()
        idx = np.where(high_marks, np.arange(df.height), -1)
        last_idx = np.maximum.accumulate(idx)

        last_high_inclusive = np.full(df.height, np.nan, dtype=np.float64)
        valid = last_idx >= 0
        last_high_inclusive[valid] = diff[last_idx[valid]]

        prev_high = np.roll(last_high_inclusive, 1)
        prev_high[0] = np.nan

        index = 1 - (diff / prev_high)
        index[index < 0] = 0

        x = df.get_column('Date').to_numpy()
        y_out = np.nan_to_num(index, nan=0.0)

        ax[0].set_title(self.description)
        ax[0].set_xlabel('Date')
        ax[0].set_ylabel('PiCycleIndex')
        sns.lineplot(x=x, y=y_out, ax=ax[0])

        ax[1].set_xlabel('Date')
        ax[1].set_ylabel('PiCycleDiff')
        sns.lineplot(x=x, y=df.get_column('PiCycleDiff').to_numpy(), ax=ax[1])
        sns.lineplot(
            x=x,
            y=df.get_column('PiCycleDiffThreshold').to_numpy(),
            ax=ax[1],
            linestyle='--',
        )

        return pl.Series('PiCycleIndex', index)
