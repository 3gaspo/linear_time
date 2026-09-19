"""NumPy source adaptation of DLinear's endpoint-padded moving average.

Upstream: https://github.com/cure-lab/LTSF-Linear
Pinned revision: 0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6.
Reference adaptation: TimeTensors external_models/dlinear/model.py.
The padding/decomposition equations are preserved; pooling is implemented in
NumPy and operates on channel-independent (window, time) inputs. Forecast
heads and closed-form fitting belong to this project's proposal, not upstream.
"""

import numpy as np


def moving_average(x: np.ndarray, kernel_size: int = 25) -> np.ndarray:
    if kernel_size < 1 or kernel_size % 2 != 1:
        raise ValueError("DLinear moving-average kernel must be a positive odd integer")
    pad = (kernel_size - 1) // 2
    array = np.asarray(x, dtype=np.float64)
    padded = np.pad(array, [(0, 0)] * (array.ndim - 1) + [(pad, pad)], mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel_size, axis=-1)
    return windows.mean(axis=-1, dtype=np.float64)


def decompose(x: np.ndarray, kernel_size: int = 25) -> tuple[np.ndarray, np.ndarray]:
    trend = moving_average(x, kernel_size)
    return x - trend, trend
