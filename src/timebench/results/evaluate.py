"""Panel evaluation for fitted ridge and non-learned references."""

from time import perf_counter
from typing import Iterable

import numpy as np

from timebench.data.windows import WindowBatch
from timebench.proposal.ridge import FittedRidge
from .metrics import METRICS, compute_window_metrics, summarize_metrics


def validation_score(batches: Iterable[WindowBatch], model: FittedRidge,
    metric: str, epsilon: float) -> float:
    total, count = 0.0, 0
    for batch in batches:
        values = compute_window_metrics(model.predict(batch), batch.y, batch.x,
            batch.mase_scales, epsilon=epsilon)[metric]
        finite = values[np.isfinite(values)]
        total += float(finite.sum())
        count += len(finite)
    if not count:
        raise ValueError("Alpha selection has no finite validation losses")
    return total / count


def evaluate_batches(batches: Iterable[WindowBatch], model: FittedRidge, *, epsilon: float):
    methods = ("ridge", "persistence", "seasonal_naive")
    losses = {method: {metric: [] for metric in METRICS} for method in methods}
    metadata = {name: [] for name in ("member_ids", "origins", "constant")}
    timings = {method: 0.0 for method in methods}
    finite_targets = 0
    window_count = 0
    for batch in batches:
        B, N, _ = batch.y.shape
        window_count += B
        metadata["member_ids"].append(np.tile(batch.member_ids, B))
        metadata["origins"].append(np.repeat(batch.origins, N))
        metadata["constant"].append(batch.constant.reshape(-1))
        finite_targets += int(np.isfinite(batch.y).sum())
        for method in methods:
            started = perf_counter()
            if method == "ridge":
                prediction = model.predict(batch)
            elif method == "persistence":
                prediction = np.repeat(batch.x[:, :, -1:], model.H, axis=2)
            else:
                if batch.seasonal_prediction is None:
                    raise ValueError("Evaluation requires Seasonal Naive")
                prediction = batch.seasonal_prediction
            timings[method] += (batch.seasonal_prediction_seconds if method == "seasonal_naive"
                else perf_counter() - started)
            values = compute_window_metrics(prediction, batch.y, batch.x,
                batch.mase_scales, epsilon=epsilon)
            for metric in METRICS:
                losses[method][metric].append(values[metric])
    arrays = {
        name: np.concatenate(parts) if parts else np.array([])
        for name, parts in metadata.items()
    }
    summaries = {}
    for method in methods:
        metrics = {name: np.concatenate(parts, axis=0) if parts else np.empty((0, 0))
            for name, parts in losses[method].items()}
        summaries[method] = {"status": "evaluated" if len(arrays["member_ids"]) else "no_windows",
            "metrics": summarize_metrics(metrics, arrays["member_ids"])}
        arrays.update({f"{method}.{name}": values.reshape(-1) for name, values in metrics.items()})
    seasonal = summaries["seasonal_naive"]["metrics"]["mase"]
    denominator = seasonal["mean"]
    for method in methods:
        metrics = summaries[method]["metrics"]
        for name in ("mase", "w10_mase"):
            raw = metrics[name]["mean"]
            available = denominator is not None and denominator > 0 and raw is not None
            scaled = {"mean": raw / denominator if available else None, "seasonal_mean": denominator}
            if name == "mase":
                scaled["std"] = metrics[name]["std"] / denominator if available else None
                scaled["variance"] = metrics[name]["variance"] / denominator ** 2 if available else None
            metrics["scaled_mase" if name == "mase" else "w10_scaled_mase"] = scaled
        summaries[method]["seasonal_mase_variance"] = seasonal["variance"]
    return summaries, arrays, {"prediction_seconds": timings,
        "windows": window_count,
        "finite_targets": finite_targets, "device": "cpu"}
