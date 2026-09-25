"""Deterministic multivariate TIME windows streamed in bounded batches."""

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Iterator, Sequence

import numpy as np

from .splits import time_intervals
from .time import TimePanel


@dataclass(frozen=True)
class WindowBatch:
    x: np.ndarray
    y: np.ndarray
    member_ids: np.ndarray
    panel_id: str
    origins: np.ndarray
    mase_scales: np.ndarray
    constant: np.ndarray
    seasonal_prediction: np.ndarray | None = None
    seasonal_prediction_seconds: float = 0.0

    def select(self, mask: np.ndarray) -> "WindowBatch":
        return replace(self, **{name: getattr(self, name)[mask]
            for name in ("x", "y", "origins", "mase_scales", "constant")},
            seasonal_prediction=(self.seasonal_prediction[mask]
                if self.seasonal_prediction is not None else None))


def fill_missing_history(context: np.ndarray) -> np.ndarray:
    values = np.asarray(context, dtype=np.float64).copy()
    for history in values.reshape(-1, values.shape[-1]):
        missing = np.isnan(history)
        if not missing.any():
            continue
        indexes = np.where(~missing, np.arange(len(history)), 0)
        np.maximum.accumulate(indexes, out=indexes)
        history[:] = history[indexes]
        observed = history[np.isfinite(history)]
        history[np.isnan(history)] = observed[0] if observed.size else 0.0
    return values


def seasonal_scale_prefix(values: np.ndarray, seasonality: int) -> tuple[np.ndarray, np.ndarray]:
    if seasonality <= 0:
        raise ValueError("seasonality must be positive")
    differences = np.zeros(len(values), dtype=np.float64)
    counts = np.zeros(len(values), dtype=np.int64)
    if len(values) > seasonality:
        left, right = values[:-seasonality], values[seasonality:]
        valid = np.isfinite(left) & np.isfinite(right)
        differences[seasonality:] = np.where(valid, np.abs(right - left), 0.0)
        counts[seasonality:] = valid
    return np.r_[0.0, np.cumsum(differences)], np.r_[0, np.cumsum(counts)]


def maximum_test_context(panel: TimePanel, test_length: int) -> int:
    return panel.values.shape[1] - test_length


def validation_origin_counts(length: int, *, L: int, H: int,
    val_length: int, test_length: int) -> dict[str, int]:
    test_start = length - test_length
    test_dates = len(np.arange(test_start, length - H + 1, H, dtype=np.int64))
    requested = min(test_dates, val_length // H)
    candidates = test_start - H * np.arange(requested, 0, -1, dtype=np.int64)
    available = int((candidates >= L).sum())
    return {"requested": int(requested), "available": available}


def iter_window_batches(
    panel: TimePanel, *, members: Sequence[str] | None, split: str,
    L: int, H: int, val_length: int, test_length: int,
    stride: int | None = None, batch_size: int = 512,
    remove_constants: bool = False, constant_epsilon: float = 1e-8,
) -> Iterator[WindowBatch]:
    if min(L, H, batch_size) <= 0 or constant_epsilon < 0:
        raise ValueError("L, H, batch_size must be positive; epsilon nonnegative")
    wanted = set(panel.member_ids if members is None else members)
    indexes = np.asarray([i for i, member in enumerate(panel.member_ids) if member in wanted], dtype=int)
    if len(indexes) != len(wanted):
        raise ValueError("Selected panel members are unavailable")
    values = panel.values[indexes]
    member_ids = np.asarray(panel.member_ids)[indexes]
    length = values.shape[1]
    lower, upper = time_intervals(length, val_length, test_length)[split]
    if split != "train" and upper - lower < H:
        raise ValueError(f"H={H} does not fit TIME's {split} interval")
    if split == "valid":
        counts = validation_origin_counts(length, L=L, H=H,
            val_length=val_length, test_length=test_length)
        test_start = length - test_length
        origins = test_start - H * np.arange(counts["requested"], 0, -1, dtype=np.int64)
        origins = origins[origins >= L]
    else:
        step = (1 if split == "train" else H) if stride is None else int(stride)
        origins = np.arange(max(L, lower), upper - H + 1, step, dtype=np.int64)
    if split == "test" and not len(origins):
        raise ValueError(f"No complete {split} window for L={L}, H={H} in {panel.panel_id}")
    prefixes = [seasonal_scale_prefix(row, panel.seasonality) for row in values]
    full_history = fill_missing_history(values)
    for start in range(0, len(origins), batch_size):
        query = origins[start:start + batch_size]
        raw_x = values[:, query[:, None] - L + np.arange(L)].transpose(1, 0, 2)
        y = values[:, query[:, None] + np.arange(H)].transpose(1, 0, 2)
        eligible = np.isfinite(raw_x).any(axis=2).all(axis=1)
        eligible &= (np.isfinite(y).all(axis=(1, 2)) if split == "train"
            else np.isfinite(y).any(axis=(1, 2)))
        x = fill_missing_history(raw_x)
        constant = x.std(axis=2, ddof=0) <= constant_epsilon
        if remove_constants:
            eligible &= ~constant.any(axis=1)
        scales = np.column_stack([np.divide(
            sums[query], counts[query], out=np.full(len(query), np.nan), where=counts[query] > 0)
            for sums, counts in prefixes])
        scales = np.where(scales > 0, scales, np.nan)
        batch = WindowBatch(x, y, member_ids, panel.panel_id, query, scales, constant).select(eligible)
        if len(batch.x):
            started = perf_counter()
            periods = np.minimum(panel.seasonality, batch.origins)
            positions = batch.origins[:, None] - periods[:, None] + np.arange(H)[None, :] % periods[:, None]
            prediction = np.stack([row[positions] for row in full_history], axis=1)
            yield replace(batch, seasonal_prediction=prediction,
                seasonal_prediction_seconds=perf_counter() - started)
