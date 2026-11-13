"""Preprocessing helpers for GRACE-FO and Sentinel-2 grids."""

import numpy as np


def resample_to_grid(values: np.ndarray, target_shape: tuple) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError("expected a two-dimensional grid")
    rows = np.linspace(0, values.shape[0] - 1, target_shape[0])
    cols = np.linspace(0, values.shape[1] - 1, target_shape[1])
    row_idx = np.round(rows).astype(int)
    col_idx = np.round(cols).astype(int)
    return values[np.ix_(row_idx, col_idx)]


def normalise_units(values: np.ndarray, scale: float) -> np.ndarray:
    return values * scale
