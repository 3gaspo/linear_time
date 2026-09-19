"""Flat task rows and explicit seed reductions for Linear TIME reports."""

import json
from pathlib import Path

import numpy as np

from timebench.pipeline.runs import load_manifest
from .metrics import METRICS

REPORT_METRICS = (*METRICS, *(f"w10_{name}" for name in METRICS), "scaled_mase", "w10_scaled_mase")


def load_performance_rows(run_dirs):
    rows = []
    for run in map(Path, run_dirs):
        manifest = load_manifest(run)
        if manifest["project"] != "linear_time" or manifest["status"] != "completed":
            raise ValueError("Only completed Linear TIME tasks may enter reports")
        summary = json.loads((run / "metrics_summary.json").read_text())
        timing = json.loads((run / "timing.json").read_text())
        selection = json.loads((run / "alpha_selection.json").read_text())
        for population, methods in summary.items():
            for method, payload in methods.items():
                if payload["status"] != "evaluated":
                    continue
                row = {**manifest["identity"], "model": method,
                    "population": population, "study": manifest["pipeline_config"]["study"],
                    "seed": manifest["pipeline_config"].get("seed"),
                    "selected_alpha": selection["selected_alpha"],
                    "inference_seconds": timing["groups"][population]["prediction_seconds"][method],
                    "fit_seconds": timing["fit_seconds"] if method == "ridge" else 0.0,
                    "alpha_selection_seconds": timing["alpha_selection_seconds"] if method == "ridge" else 0.0,
                    "seasonal_mase_variance": payload["seasonal_mase_variance"]}
                for metric in METRICS:
                    row[metric] = payload["metrics"][metric]["mean"]
                    row[f"{metric}_std"] = payload["metrics"][metric]["std"]
                    row[f"{metric}_variance"] = payload["metrics"][metric]["variance"]
                    row[f"{metric}_user_mean"] = payload["metrics"][metric]["user_mean"]
                    row[f"w10_{metric}"] = payload["metrics"][f"w10_{metric}"]["mean"]
                row["scaled_mase"] = payload["metrics"]["scaled_mase"]["mean"]
                row["w10_scaled_mase"] = payload["metrics"]["w10_scaled_mase"]["mean"]
                rows.append(row)
    return rows


def seed_statistics(values):
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    complete = bool(values) and len(finite) == len(values)
    mean = float(np.mean(finite)) if complete else None
    std = float(np.std(finite, ddof=1)) if complete and len(finite) > 1 else None
    return {"mean": mean, "seed_std": std, "seed_count": len(values),
        "finite_seeds": len(finite), "lower": mean - std if std is not None else None,
        "upper": mean + std if std is not None else None, "seed_dispersion_ddof": 1}


def aggregate_seed_rows(rows, seeds):
    keys = ("dataset", "panel", "structure", "members", "L_term", "H_term", "L", "H", "mode", "model", "population")
    groups = {}
    for row in rows:
        group = groups.setdefault(tuple(row[key] for key in keys), {})
        if row["seed"] in group:
            raise ValueError("Duplicate task/seed report row")
        group[row["seed"]] = row
    result = []
    for key, group in groups.items():
        if set(group) != set(seeds):
            raise ValueError(f"Missing requested user-split seed: {key}")
        repeated = [group[seed] for seed in seeds]
        row = {name: value for name, value in zip(keys, key)}
        row["study"] = repeated[0]["study"]
        for metric in REPORT_METRICS:
            stats = seed_statistics([item[metric] for item in repeated])
            row[metric] = stats["mean"]
            row[f"{metric}_seed_std"] = stats["seed_std"]
            row[f"{metric}_lower"] = stats["lower"]
            row[f"{metric}_upper"] = stats["upper"]
        for field in [*(f"{metric}_{suffix}" for metric in METRICS
                for suffix in ("std", "variance", "user_mean")),
                "seasonal_mase_variance", "inference_seconds", "fit_seconds", "alpha_selection_seconds"]:
            row[field] = seed_statistics([item[field] for item in repeated])["mean"]
        row["seed_count"] = len(seeds)
        row["selected_alphas"] = [item["selected_alpha"] for item in repeated]
        result.append(row)
    return result
