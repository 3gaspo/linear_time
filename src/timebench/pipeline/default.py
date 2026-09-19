"""Compose the three Linear TIME experiment families."""

import os

from timebench.data.grid import DEFAULT_EXCLUDED, load_grid
from timebench.data.time import load_dataset_semantics, load_time_panels
from .tasks import run_panel_task, task_root, write_json


def study_datasets(config):
    datasets = [name for name in load_grid() if name not in set(config["excluded_datasets"])]
    if config["experiment_mode"] == "test":
        return ["SG_PM25/H"]
    if config["experiment_mode"] != "full":
        raise ValueError("experiment_mode must be test or full")
    return datasets


def run_study(config):
    study = config["study"]
    if study not in {"default", "user_generalization", "variate_modes"}:
        raise ValueError("Unknown study")
    seeds = [int(seed) for seed in config["user_generalization"]["seeds"]]
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("User-generalization seeds must be distinct and nonempty")
    modes = list(config["variate_modes"]["modes"])
    paths = []
    for dataset in study_datasets(config):
        semantics = load_dataset_semantics(dataset)
        panels = load_time_panels(dataset, config["data"]["storage_path"])
        if study == "user_generalization" and semantics["structure"] not in {"multiple_series", "mixed"}:
            continue
        settings = load_grid()[dataset]
        if config["experiment_mode"] == "test":
            settings = [settings[0]]
            panels = panels[:1]
        for setting in settings:
            for panel in panels:
                if study == "user_generalization":
                    if len(panel.member_ids) < 2:
                        continue
                    for seed in seeds:
                        task_config = {**config, "model": {**config["model"], "mode": "shared"}}
                        paths.append(run_panel_task(task_config, setting, panel, seed=seed))
                elif study == "variate_modes":
                    if len(panel.member_ids) < 2:
                        continue
                    for mode in modes:
                        task_config = {**config, "model": {**config["model"], "mode": mode}}
                        paths.append(run_panel_task(task_config, setting, panel))
                else:
                    task_config = {**config, "model": {**config["model"], "mode": "joint"}}
                    paths.append(run_panel_task(task_config, setting, panel))
    root = task_root(config).parent
    launch = os.environ.get("TIME_LAUNCH_ID", "manual")
    launch_root = root / "launches" / launch
    launch_root.mkdir(parents=True, exist_ok=True)
    write_json(launch_root / f"{study}_manifest.json", {
        "schema_version": 1, "experiment": config["experiment"], "study": study,
        "mode": config["experiment_mode"], "datasets": study_datasets(config),
        "seeds": seeds if study == "user_generalization" else [],
        "model_modes": modes if study == "variate_modes" else (["shared"] if study == "user_generalization" else ["joint"]),
        "task_fits": len(paths), "task_manifests": [str(path / "manifest.json") for path in paths]})
    return paths
