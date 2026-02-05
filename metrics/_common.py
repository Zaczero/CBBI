import numpy as np
import polars as pl


def join_left_on_date(df: pl.DataFrame, other: pl.DataFrame):
    return df.join(other, on='Date', how='left', maintain_order='left')


def linreg_predict(
    x_train: np.ndarray, y_train: np.ndarray, x_all: np.ndarray
) -> np.ndarray:
    x_train = x_train.astype(np.float64)
    y_train = y_train.astype(np.float64)

    x_mean = x_train.mean()
    y_mean = y_train.mean()

    x_centered = x_train - x_mean
    slope = (x_centered * (y_train - y_mean)).sum() / (x_centered * x_centered).sum()
    intercept = y_mean - slope * x_mean

    return intercept + slope * x_all.astype(np.float64)
