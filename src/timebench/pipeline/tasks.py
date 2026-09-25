"""Validation-selected panel tasks with explicit semantic and study identity."""

import json
import logging
import re
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from timebench.data.grid import WindowSetting
from timebench.data.splits import split_users
from timebench.data.time import TimePanel
from timebench.data.windows import (iter_window_batches, maximum_test_context,
    validation_origin_counts)
from timebench.paths import dataset_storage_root, outputs_root
from timebench.proposal.ridge import (RidgeConfig, load_fitted, load_problem,
    prepare_ridge, save_fitted, save_problem, solve_ridge)
from timebench.results.evaluate import evaluate_batches, validation_score
from .runs import allocate_run, manifest_reference

LOGGER = logging.getLogger(__name__)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def task_root(config: dict) -> Path:
    root = Path(config["artifacts"]["output_root"]) if config["artifacts"]["output_root"] else outputs_root()
    return root / config["experiment"] / "tasks"


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "target"


def _dependency_reference(path: Path) -> dict:
    return manifest_reference(path)


def select_alpha(problem, alphas, validation_batches, metric, epsilon, default_alpha):
    scores = []
    for alpha in alphas:
        candidate = solve_ridge(problem, alpha, calculate_condition=False)
        score = validation_score(validation_batches(), candidate, metric, epsilon)
        scores.append({"alpha": alpha, "score": score})
        LOGGER.info("Validation alpha=%s metric=%s score=%s", alpha, metric, score)
    usable = [row for row in scores if row["score"] is not None]
    if not usable:
        return float(default_alpha), scores, "no_usable_validation_windows"
    best = min(usable, key=lambda row: (row["score"], row["alpha"]))
    return best["alpha"], scores, None


def run_panel_task(config: dict, setting: WindowSetting, panel: TimePanel,
                   *, seed: int | None = None) -> Path:
    from timebench.pipeline.runtime_resources import log_selected_device
    log_selected_device('cpu', stage='fit_evaluate', component='exact_ridge')
    study, data = config["study"], config["data"]
    proposed_L = setting.L
    L = min(proposed_L, maximum_test_context(panel, setting.test_length))
    if L <= 0:
        raise ValueError(f"No context before the earliest test date for {setting.dataset}/{panel.panel_id}")
    if study == "user_generalization":
        if seed is None or len(panel.member_ids) < 2:
            raise ValueError("User generalization requires a seed and multiple members")
        partition = split_users(panel.member_ids, float(data["train_user_fraction"]), seed)
        fit_members = partition.seen
        populations = (("seen", partition.seen), ("unseen", partition.unseen))
    else:
        if seed is not None:
            raise ValueError("Seeds belong only to the user-generalization study")
        partition = None
        fit_members = panel.member_ids
        populations = (("all", panel.member_ids),)
    policy = data["constant_policy"]
    remove_train = policy in {"remove_train_windows", "remove_all_windows"}
    remove_eval = policy in {"remove_eval_windows", "remove_all_windows"}
    model_config = RidgeConfig(**config["model"])
    if study == "default" and model_config.mode != "joint":
        raise ValueError("The default experiment is the joint N*L to N*H model")
    if study == "user_generalization" and model_config.mode != "shared":
        raise ValueError("Unseen-user evaluation requires the shared independent model")
    alphas = sorted(set(float(value) for value in config["validation"]["alphas"]))
    validation = {"alphas": alphas, "metric": config["validation"]["metric"],
        "population": "fit_members_valid", "refit_validation": False}
    pipeline_config = {
        "study": study, "user_split": study == "user_generalization",
        "seed": seed, "train_user_fraction": (float(data["train_user_fraction"])
            if study == "user_generalization" else None),
        "val_length": setting.val_length, "test_length": setting.test_length,
        "train_stride": int(data["train_stride"]),
        "evaluation_stride": setting.H,
        "constant_policy": policy, "constant_epsilon": float(data["constant_epsilon"]),
        "proposed_L": proposed_L, "effective_L": L,
        "L_cap": "min(proposed_L, history available at earliest test origin)",
        "validation_origin": "walk backward from first test origin at stride H",
        "validation": validation,
    }
    identity = {"dataset": setting.dataset, "panel": panel.panel_id,
        "structure": panel.structure, "members": len(fit_members),
        "L_term": setting.L_term, "H_term": setting.H_term,
        "L": L, "H": setting.H, "model": "ridge", "mode": model_config.mode}
    scientific_model = {**asdict(model_config), "alpha": "validation"}
    root = (task_root(config) / setting.dataset / f"L={setting.L_term}_{L}"
        / f"H={setting.H_term}_{setting.H}" / f"panel={_slug(panel.panel_id)}"
        / f"mode={model_config.mode}")
    for name, value in scientific_model.items():
        if name not in {"mode", "alpha"}:
            root /= f"{name}={str(value).lower()}"
    if seed is not None:
        root /= f"seed_{seed}"
    handle = allocate_run(root, experiment=config["experiment"], identity=identity,
        model_config=scientific_model, pipeline_config=pipeline_config,
        experiment_config={"metric_epsilon": float(config["evaluation"]["epsilon"]),
            "rmse_definition": "relative_mean_squared_error",
            "scaled_mase": "matched_population_task_mean_over_seasonal_mean"},
        runtime_config={"batch_size": int(config["runtime"]["batch_size"]),
            "device": "cpu", "dtype": "float64"},
        provenance={"dataset_path": str(Path(data["storage_path"] or dataset_storage_root()) / setting.dataset),
            "improved_revision": "541a2802cd2a35d39156aef4c37de6964d112786"},
        policy=config["artifacts"]["conflict_policy"], force=bool(config["artifacts"]["force"]))
    if not handle.should_run:
        return handle.run_dir
    with handle:
        LOGGER.info("Task dataset=%s panel=%s L=%s/%s H=%s mode=%s seed=%s",
            setting.dataset, panel.panel_id, L, proposed_L, setting.H, model_config.mode, seed)
        write_json(handle.run_dir / "member_split.json", {
            "seed": seed, "fit": list(fit_members),
            "seen": list(partition.seen) if partition else list(fit_members),
            "unseen": list(partition.unseen) if partition else []})
        write_json(handle.run_dir / "panel_metadata.json", {"panel": panel.panel_id,
            "quantity": panel.quantity, "structure": panel.structure,
            "members": list(panel.member_ids), "frequency": panel.frequency,
            "seasonality": panel.seasonality, "start": panel.start,
            "length": panel.values.shape[1]})
        write_json(handle.run_dir / "config.json", {"identity": identity,
            "model": scientific_model, "pipeline": pipeline_config})
        window = {"panel": panel, "L": L, "H": setting.H,
            "val_length": setting.val_length, "test_length": setting.test_length,
            "batch_size": int(config["runtime"]["batch_size"]),
            "constant_epsilon": float(data["constant_epsilon"])}
        relative = root.relative_to(task_root(config))
        cache_base = task_root(config).parent / "cache"
        cache_runtime = {"batch_size": int(config["runtime"]["batch_size"]),
            "device": "cpu", "dtype": "float64"}
        cache_provenance = {"dataset_path": str(Path(data["storage_path"] or dataset_storage_root()) / setting.dataset),
            "parent_task_manifest": str(handle.run_dir / "manifest.json")}
        training_model = {key: value for key, value in asdict(model_config).items() if key != "alpha"}
        training_pipeline = {"stage": "training_statistics", "members": list(fit_members),
            "split": "train", "L": L, "H": setting.H,
            "val_length": setting.val_length, "test_length": setting.test_length,
            "stride": int(data["train_stride"]), "constant_policy": policy,
            "constant_epsilon": float(data["constant_epsilon"])}
        training_handle = allocate_run(cache_base / "training_statistics" / relative,
            experiment=f'{config["experiment"]}_training_statistics', identity=identity,
            model_config=training_model, pipeline_config=training_pipeline,
            experiment_config={}, runtime_config=cache_runtime,
            provenance=cache_provenance, policy=config["artifacts"]["conflict_policy"],
            force=bool(config["artifacts"]["force"]))
        if training_handle.should_run:
            with training_handle:
                started = perf_counter()
                problem = prepare_ridge(iter_window_batches(**window, members=fit_members,
                    split="train", stride=int(data["train_stride"]), remove_constants=remove_train), model_config)
                statistics_seconds = perf_counter() - started
                files = save_problem(problem, training_handle.run_dir)
                write_json(training_handle.run_dir / "timing.json", {"statistics_seconds": statistics_seconds})
                training_handle.complete([*files, "timing.json"])
        training_run = training_handle.run_dir
        problem = load_problem(training_run)
        statistics_seconds = json.loads((training_run / "timing.json").read_text(encoding="utf-8"))["statistics_seconds"]

        validation_batches = list(iter_window_batches(**window, members=fit_members,
            split="valid", stride=setting.H, remove_constants=remove_eval))
        origin_counts = validation_origin_counts(panel.values.shape[1], L=L, H=setting.H,
            val_length=setting.val_length, test_length=setting.test_length)
        validation_counts = {**origin_counts,
            "usable": int(sum(len(batch.x) for batch in validation_batches))}
        selection_pipeline = {"stage": "validation_selection",
            "training": _dependency_reference(training_run),
            "alphas": alphas, "default_alpha": model_config.alpha,
            "metric": validation["metric"], "metric_epsilon": float(config["evaluation"]["epsilon"]),
            "validation_schedule": "walk_backward_from_first_test_origin_at_stride_H",
            "validation_support": "finite_context_and_future", "counts": validation_counts,
            "constant_policy": policy, "constant_epsilon": float(data["constant_epsilon"])}
        selection_handle = allocate_run(cache_base / "validation_selection" / relative,
            experiment=f'{config["experiment"]}_validation_selection', identity=identity,
            model_config={"solver": training_model, "selection": "alpha"},
            pipeline_config=selection_pipeline, experiment_config={},
            runtime_config=cache_runtime,
            provenance={**cache_provenance, "training_manifest": str(training_run / "manifest.json")},
            policy=config["artifacts"]["conflict_policy"], force=bool(config["artifacts"]["force"]))
        if selection_handle.should_run:
            with selection_handle:
                started = perf_counter()
                selected_alpha, scores, fallback_reason = select_alpha(problem, alphas,
                    lambda: iter(validation_batches), validation["metric"],
                    float(config["evaluation"]["epsilon"]), model_config.alpha)
                selection_seconds = perf_counter() - started
                write_json(selection_handle.run_dir / "alpha_selection.json", {**validation,
                    "selected_alpha": selected_alpha, "scores": scores,
                    "fallback_reason": fallback_reason, "validation_counts": validation_counts,
                    "validation_schedule": "walk_backward_from_first_test_origin_at_stride_H",
                    "tie_rule": "smaller_alpha", "training_interval": "train_only"})
                write_json(selection_handle.run_dir / "timing.json", {"alpha_selection_seconds": selection_seconds})
                selection_handle.complete(["alpha_selection.json", "timing.json"])
        selection_run = selection_handle.run_dir
        selection_result = json.loads((selection_run / "alpha_selection.json").read_text(encoding="utf-8"))
        selection_seconds = json.loads((selection_run / "timing.json").read_text(encoding="utf-8"))["alpha_selection_seconds"]
        selected_alpha = float(selection_result["selected_alpha"])

        coefficient_pipeline = {"stage": "fitted_coefficients",
            "training": _dependency_reference(training_run), "selected_alpha": selected_alpha}
        coefficient_handle = allocate_run(cache_base / "fitted_coefficients" / relative,
            experiment=f'{config["experiment"]}_fitted_coefficients', identity=identity,
            model_config={**training_model, "alpha": selected_alpha},
            pipeline_config=coefficient_pipeline, experiment_config={},
            runtime_config=cache_runtime,
            provenance={**cache_provenance, "selection_manifest": str(selection_run / "manifest.json")},
            policy=config["artifacts"]["conflict_policy"], force=bool(config["artifacts"]["force"]))
        if coefficient_handle.should_run:
            with coefficient_handle:
                started = perf_counter()
                model = solve_ridge(problem, selected_alpha)
                solve_seconds = perf_counter() - started
                files = save_fitted(model, coefficient_handle.run_dir)
                write_json(coefficient_handle.run_dir / "timing.json", {"selected_solve_seconds": solve_seconds})
                coefficient_handle.complete([*files, "timing.json"])
        coefficient_run = coefficient_handle.run_dir
        model = load_fitted(coefficient_run)
        solve_seconds = json.loads((coefficient_run / "timing.json").read_text(encoding="utf-8"))["selected_solve_seconds"]

        write_json(handle.run_dir / "alpha_selection.json", selection_result)
        write_json(handle.run_dir / "fit_summary.json", model.metadata())
        names = list(model.coefficients)
        arrays = {f"head_{i}": model.coefficients[name] for i, name in enumerate(names)}
        if model.joint_dual is not None:
            arrays.update(joint_train_features=model.joint_train_features,
                joint_dual=model.joint_dual, joint_feature_mean=model.joint_feature_mean,
                joint_target_mean=model.joint_target_mean)
        np.savez_compressed(handle.run_dir / "coefficients.npz", **arrays)
        write_json(handle.run_dir / "coefficient_heads.json", {
            "representation": "exact_dual" if model.joint_dual is not None else "exact_primal",
            "heads": {f"head_{i}": name for i, name in enumerate(names)}})
        summaries, payloads = {}, {}
        timing = {"fit_seconds": statistics_seconds + solve_seconds,
            "statistics_seconds": statistics_seconds, "selected_solve_seconds": solve_seconds,
            "alpha_selection_seconds": selection_seconds, "device": "cpu", "groups": {}}
        for population, members in populations:
            if not members:
                continue
            for split in ("valid", "test"):
                group = f"{population}_{split}"
                evaluation_pipeline = {"stage": "evaluation", "population": population,
                    "split": split, "members": list(members),
                    "coefficients": _dependency_reference(coefficient_run),
                    "stride": setting.H, "constant_policy": policy,
                    "constant_epsilon": float(data["constant_epsilon"]),
                    "metric_epsilon": float(config["evaluation"]["epsilon"])}
                evaluation_handle = allocate_run(
                    cache_base / "evaluation" / population / split / relative,
                    experiment=f'{config["experiment"]}_evaluation', identity=identity,
                    model_config={**training_model, "alpha": selected_alpha},
                    pipeline_config=evaluation_pipeline, experiment_config={},
                    runtime_config=cache_runtime,
                    provenance={**cache_provenance, "coefficient_manifest": str(coefficient_run / "manifest.json")},
                    policy=config["artifacts"]["conflict_policy"], force=bool(config["artifacts"]["force"]))
                if evaluation_handle.should_run:
                    with evaluation_handle:
                        summary, evaluation_arrays, group_timing = evaluate_batches(
                            iter_window_batches(**window, members=members, split=split,
                                stride=setting.H, remove_constants=remove_eval),
                            model, epsilon=float(config["evaluation"]["epsilon"]))
                        write_json(evaluation_handle.run_dir / "metrics_summary.json", summary)
                        np.savez_compressed(evaluation_handle.run_dir / "window_metrics.npz", **evaluation_arrays)
                        write_json(evaluation_handle.run_dir / "timing.json", group_timing)
                        evaluation_handle.complete(["metrics_summary.json", "window_metrics.npz", "timing.json"])
                evaluation_run = evaluation_handle.run_dir
                summary = json.loads((evaluation_run / "metrics_summary.json").read_text(encoding="utf-8"))
                group_timing = json.loads((evaluation_run / "timing.json").read_text(encoding="utf-8"))
                with np.load(evaluation_run / "window_metrics.npz", allow_pickle=False) as saved:
                    evaluation_arrays = {name: saved[name] for name in saved.files}
                summaries[group] = summary
                payloads.update({f"{group}.{name}": values for name, values in evaluation_arrays.items()})
                timing["groups"][group] = group_timing
        write_json(handle.run_dir / "metrics_summary.json", summaries)
        write_json(handle.run_dir / "timing.json", timing)
        np.savez_compressed(handle.run_dir / "window_metrics.npz", **payloads)
        handle.complete(["config.json", "member_split.json", "panel_metadata.json",
            "alpha_selection.json", "fit_summary.json", "coefficients.npz",
            "coefficient_heads.json", "metrics_summary.json", "timing.json", "window_metrics.npz"])
    return handle.run_dir
