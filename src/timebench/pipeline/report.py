"""Exact-study reports with optional user-split seed intervals."""

import json
import os
from pathlib import Path

import pandas as pd

from timebench.results.performance import write_table
from timebench.results.reporting import REPORT_METRICS, aggregate_seed_rows, load_performance_rows
from timebench.visualization.linear import plot_lh_heatmap, plot_seed_intervals
from .runs import load_manifest
from .tasks import _slug, task_root, write_json


def _launch_runs(config, launch_id):
    path = task_root(config).parent / "launches" / launch_id / f"{config['study']}_manifest.json"
    if not path.is_file():
        raise ValueError(f"Missing exact launch manifest: {path}")
    launch = json.loads(path.read_text(encoding="utf-8"))
    if launch.get("study") != config["study"]:
        raise ValueError(f"Launch {launch_id} belongs to {launch.get('study')}")
    selected = []
    for manifest_path in launch["task_manifests"]:
        manifest_path = Path(manifest_path)
        manifest = load_manifest(manifest_path)
        if manifest["status"] != "completed":
            raise ValueError(f"Launch task is not completed: {manifest_path}")
        if manifest["pipeline_config"].get("study") != config["study"]:
            raise ValueError(f"Launch task has the wrong study: {manifest_path}")
        selected.append((manifest_path.parent, manifest))
    if len(selected) != launch["task_fits"]:
        raise ValueError(f"Launch {launch_id} expected {launch['task_fits']} tasks, found {len(selected)}")
    return selected, path


def write_study_report(config):
    from timebench.pipeline.runtime_resources import log_selected_device
    log_selected_device('cpu', stage='report', component='linear_time')
    study = config["study"]
    report_id = (config["report"]["report_id"] or os.environ.get("TIME_LAUNCH_ID")
        or "manual")
    selected, launch_manifest = _launch_runs(config, report_id)
    if not selected:
        raise ValueError(f"No completed {study} tasks")
    rows = load_performance_rows([path for path, _ in selected])
    seeds = [int(seed) for seed in config["user_generalization"]["seeds"]]
    reduced = aggregate_seed_rows(rows, seeds) if study == "user_generalization" else rows
    root = task_root(config).parents[1] / "reports" / study / report_id
    root.mkdir(parents=True, exist_ok=True)
    artifacts = write_table(pd.DataFrame(rows), root / "task_runs")
    if study == "user_generalization":
        artifacts.extend(write_table(pd.DataFrame(reduced), root / "task_seed_summary"))
    if config["report"]["plots"]:
        keys = sorted({(row["dataset"], row["panel"], row["mode"], row["model"], row["population"])
            for row in reduced})
        for dataset, panel, mode, model, population in keys:
            part = [row for row in reduced if (row["dataset"], row["panel"], row["mode"],
                row["model"], row["population"]) == (dataset, panel, mode, model, population)]
            destination = root / "performance" / population / dataset / _slug(str(panel)) / mode / model
            for metric in REPORT_METRICS:
                if any(row.get(metric) is not None for row in part):
                    artifacts.extend(plot_lh_heatmap(part, metric=metric,
                        destination=destination / metric))
                if study == "user_generalization" and any(
                    row.get(f"{metric}_seed_std") is not None for row in part):
                    artifacts.extend(plot_seed_intervals(part, metric=metric,
                        destination=destination / f"{metric}_seed_intervals"))
    write_json(root / "report_manifest.json", {"schema_version": 1,
        "project": "linear_time", "study": study,
        "launch_manifest": str(launch_manifest),
        "task_runs": len(selected), "seeds": seeds if study == "user_generalization" else [],
        "seed_std": "sample SD across user-partition seeds only, ddof=1" if study == "user_generalization" else None,
        "input_manifests": [str(path / "manifest.json") for path, _ in selected],
        "artifacts": [str(path.relative_to(root)) for path in artifacts]})
    return root
