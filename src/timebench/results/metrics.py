"""Five point metrics and W10 over per-user mean errors.

`rmse` is relative MSE; it never denotes a square-root metric here.
Window summaries and dispersion use finite window-level metric cells, ddof=0.
"""

import math

import numpy as np

METRICS = ("mase", "mae", "mse", "nmse", "rmse")


def compute_window_metrics(
    prediction: np.ndarray, target: np.ndarray, context: np.ndarray,
    mase_scales: np.ndarray, *, epsilon: float = 1e-8,
) -> dict[str, np.ndarray]:
    if epsilon <= 0:
        raise ValueError("Metric epsilon must be positive")
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != target.shape or target.ndim != 3 or context.shape[:2] != target.shape[:2]:
        raise ValueError("Prediction/target shapes and input rows must align")
    valid = np.isfinite(target)
    if not np.isfinite(prediction[valid]).all():
        raise ValueError("Non-finite prediction on a finite target")
    counts = valid.sum(axis=2)
    error = np.where(valid, prediction - target, 0.0)
    mae = np.divide(np.abs(error).sum(axis=2), counts, out=np.full(counts.shape, np.nan), where=counts > 0)
    mse = np.divide(np.square(error).sum(axis=2), counts, out=np.full(counts.shape, np.nan), where=counts > 0)
    return {
        "mase": mae / np.asarray(mase_scales),
        "mae": mae, "mse": mse,
        "nmse": mse / np.square(context.std(axis=2, ddof=0) + epsilon),
        "rmse": mse / np.square(np.abs(context.mean(axis=2)) + epsilon),
    }


def summarize_metrics(metrics: dict[str, np.ndarray], user_ids: np.ndarray) -> dict[str, dict]:
    """W10 averages the largest ceil(10% * finite users) user means.

    Keep its user count and selected identities explicit. All metrics use
    their own finite support, which can differ for MASE when no pairs exist.
    """
    result = {}
    for name in METRICS:
        values = np.asarray(metrics[name], dtype=np.float64).reshape(-1)
        if len(values) != len(user_ids):
            raise ValueError("Metric rows and user IDs must align")
        finite = np.isfinite(values)
        cells = values[finite]
        users = np.unique(user_ids[finite])
        user_means = np.asarray([values[(user_ids == user) & finite].mean() for user in users])
        result[name] = {
            "mean": float(cells.mean()) if cells.size else None,
            "std": float(cells.std(ddof=0)) if cells.size else None,
            "variance": float(cells.var(ddof=0)) if cells.size else None,
            "dispersion_ddof": 0, "finite_values": int(cells.size),
            "total_values": int(values.size), "finite_users": int(len(users)),
            "user_mean": float(user_means.mean()) if len(users) else None,
            "user_std": float(user_means.std(ddof=0)) if len(users) else None,
        }
        tail_count = math.ceil(0.1 * len(users))
        # Stable ordering makes equal-error user identities reproducible.
        tail = np.argsort(-user_means, kind="stable")[:tail_count]
        result[f"w10_{name}"] = {
            "mean": float(user_means[tail].mean()) if tail_count else None,
            "finite_users": int(len(users)), "tail_users": tail_count,
            "user_ids": users[tail].tolist(),
        }
    return result
