"""Exact weighted ridge for joint, independent, and shared panel modes."""

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from timebench.data.windows import WindowBatch
from timebench.external_models.dlinear.decomposition import decompose


@dataclass(frozen=True)
class RidgeConfig:
    loss: str = "mse"
    normalization: str = "none"
    detrend: str = "none"
    kernel_size: int = 25
    mode: str = "joint"
    alpha: float = 1.0
    intercept: bool = True
    repeat_constant: bool = False
    epsilon: float = 1e-8
    solver_rcond: float = 1e-12

    def __post_init__(self) -> None:
        if self.loss not in {"mse", "nmse", "rmse"}:
            raise ValueError("Ridge loss must be mse, nmse, or rmse (relative MSE)")
        if self.normalization not in {"none", "center", "zscore", "relative_mean"}:
            raise ValueError("Unknown normalization")
        if self.detrend not in {"none", "moving_average"}:
            raise ValueError("Unknown decomposition")
        if self.mode not in {"joint", "independent", "shared"}:
            raise ValueError("mode must be joint, independent, or shared")
        if self.alpha < 0 or self.epsilon <= 0 or self.solver_rcond <= 0:
            raise ValueError("alpha must be nonnegative; epsilon and solver_rcond positive")
        if self.kernel_size < 1 or self.kernel_size % 2 != 1:
            raise ValueError("Moving-average kernel must be positive and odd")


def context_statistics(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return x.mean(axis=2, keepdims=True), x.std(axis=2, ddof=0, keepdims=True)


def loss_denominator(x: np.ndarray, loss: str, epsilon: float) -> np.ndarray:
    mean, std = context_statistics(x)
    if loss == "mse":
        return np.ones(x.shape[:2], dtype=np.float64)
    if loss == "nmse":
        return std[:, :, 0] + epsilon
    if loss == "rmse":
        return np.abs(mean[:, :, 0]) + epsilon
    raise ValueError(f"Unknown loss {loss!r}")


def prepared_channels(x: np.ndarray, config: RidgeConfig):
    mean, std = context_statistics(x)
    offset = mean if config.normalization in {"center", "zscore"} else np.zeros_like(mean)
    if config.normalization == "zscore":
        scale = std + config.epsilon
    elif config.normalization == "relative_mean":
        scale = np.abs(mean) + config.epsilon
    else:
        scale = np.ones_like(mean)
    features = (x - offset) / scale
    if config.detrend == "moving_average":
        residual, trend = decompose(features, config.kernel_size)
        features = np.concatenate((residual, trend), axis=2)
    return features, offset, scale


def _with_intercept(features: np.ndarray, intercept: bool) -> np.ndarray:
    return np.column_stack((features, np.ones(len(features)))) if intercept else features


@dataclass
class _Statistics:
    gram: np.ndarray
    rhs: np.ndarray
    windows: int = 0
    fitted_cells: int = 0
    bypassed_cells: int = 0


@dataclass
class RidgeProblem:
    config: RidgeConfig
    L: int
    H: int
    member_ids: tuple[str, ...]
    statistics: dict[str, _Statistics]
    joint_features: np.ndarray | None = None
    joint_targets: np.ndarray | None = None
    joint_panel_id: str | None = None


@dataclass
class FittedRidge:
    config: RidgeConfig
    L: int
    H: int
    member_ids: tuple[str, ...]
    coefficients: dict[str, np.ndarray]
    diagnostics: dict[str, dict]
    joint_train_features: np.ndarray | None = None
    joint_dual: np.ndarray | None = None
    joint_feature_mean: np.ndarray | None = None
    joint_target_mean: np.ndarray | None = None

    def predict(self, batch: WindowBatch) -> np.ndarray:
        if batch.x.shape[2] != self.L:
            raise ValueError("Prediction L differs from fitted L")
        features, offset, scale = prepared_channels(batch.x, self.config)
        B, N, _ = features.shape
        result = np.empty((B, N, self.H), dtype=np.float64)
        if self.config.mode == "joint":
            if tuple(batch.member_ids) != self.member_ids:
                raise ValueError("Joint ridge requires the fitted member order")
            z = features.reshape(B, -1)
            if self.joint_dual is not None:
                centered = z - self.joint_feature_mean
                raw = ((centered @ self.joint_train_features.T / len(self.joint_train_features))
                    @ self.joint_dual + self.joint_target_mean).reshape(B, N, self.H)
            else:
                z = _with_intercept(z, self.config.intercept)
                raw = (z @ self.coefficients[batch.panel_id]).reshape(B, N, self.H)
            result[:] = raw * scale + offset
        elif self.config.mode == "shared":
            z = _with_intercept(features.reshape(B * N, -1), self.config.intercept)
            raw = (z @ self.coefficients["shared"]).reshape(B, N, self.H)
            result[:] = raw * scale + offset
        else:
            for index, member in enumerate(batch.member_ids):
                if member not in self.coefficients:
                    raise ValueError(f"No independent ridge head for {member}")
                z = _with_intercept(features[:, index], self.config.intercept)
                result[:, index] = (z @ self.coefficients[str(member)]) * scale[:, index] + offset[:, index]
        if self.config.repeat_constant:
            result = np.where(batch.constant[:, :, None], batch.x[:, :, -1:], result)
        return result

    def metadata(self) -> dict:
        return {"config": asdict(self.config), "L": self.L, "H": self.H,
            "members": list(self.member_ids), "heads": self.diagnostics,
            "representation": "exact_dual" if self.joint_dual is not None else "exact_primal",
            "lag_order": "member-major, oldest-to-newest",
            "output_order": "member-major, horizon-within-member"}


def save_problem(problem: RidgeProblem, destination: str | Path) -> list[str]:
    """Persist reusable training statistics without solving a selected alpha."""
    destination = Path(destination)
    arrays = {}
    states = {}
    for index, (name, state) in enumerate(problem.statistics.items()):
        arrays[f"gram_{index}"] = state.gram
        arrays[f"rhs_{index}"] = state.rhs
        states[str(index)] = {"name": name, "windows": state.windows,
            "fitted_cells": state.fitted_cells, "bypassed_cells": state.bypassed_cells}
    if problem.joint_features is not None:
        arrays["joint_features"] = problem.joint_features
        arrays["joint_targets"] = problem.joint_targets
    np.savez_compressed(destination / "training_statistics.npz", **arrays)
    (destination / "training_statistics.json").write_text(json.dumps({
        "schema_version": 1, "config": asdict(problem.config),
        "L": problem.L, "H": problem.H, "member_ids": list(problem.member_ids),
        "states": states, "joint_panel_id": problem.joint_panel_id,
        "representation": "joint_windows" if problem.joint_features is not None else "sufficient_statistics",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ["training_statistics.npz", "training_statistics.json"]


def load_problem(source: str | Path) -> RidgeProblem:
    source = Path(source)
    metadata = json.loads((source / "training_statistics.json").read_text(encoding="utf-8"))
    arrays = np.load(source / "training_statistics.npz", allow_pickle=False)
    statistics = {}
    for index, state in metadata["states"].items():
        statistics[state["name"]] = _Statistics(
            arrays[f"gram_{index}"], arrays[f"rhs_{index}"],
            int(state["windows"]), int(state["fitted_cells"]), int(state["bypassed_cells"]),
        )
    joint_features = arrays["joint_features"] if "joint_features" in arrays.files else None
    joint_targets = arrays["joint_targets"] if "joint_targets" in arrays.files else None
    return RidgeProblem(
        RidgeConfig(**metadata["config"]), int(metadata["L"]), int(metadata["H"]),
        tuple(metadata["member_ids"]), statistics, joint_features, joint_targets,
        metadata.get("joint_panel_id"),
    )


def save_fitted(model: FittedRidge, destination: str | Path) -> list[str]:
    destination = Path(destination)
    names = list(model.coefficients)
    arrays = {f"head_{index}": model.coefficients[name] for index, name in enumerate(names)}
    for name in ("joint_train_features", "joint_dual", "joint_feature_mean", "joint_target_mean"):
        value = getattr(model, name)
        if value is not None:
            arrays[name] = value
    np.savez_compressed(destination / "coefficients.npz", **arrays)
    (destination / "coefficient_heads.json").write_text(json.dumps({
        "schema_version": 1, "metadata": model.metadata(),
        "heads": {f"head_{index}": name for index, name in enumerate(names)},
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ["coefficients.npz", "coefficient_heads.json"]


def load_fitted(source: str | Path) -> FittedRidge:
    source = Path(source)
    metadata = json.loads((source / "coefficient_heads.json").read_text(encoding="utf-8"))
    model_metadata = metadata["metadata"]
    arrays = np.load(source / "coefficients.npz", allow_pickle=False)
    coefficients = {name: arrays[key] for key, name in metadata["heads"].items()}
    optional = {name: arrays[name] if name in arrays.files else None for name in
        ("joint_train_features", "joint_dual", "joint_feature_mean", "joint_target_mean")}
    return FittedRidge(
        RidgeConfig(**model_metadata["config"]), int(model_metadata["L"]),
        int(model_metadata["H"]), tuple(model_metadata["members"]), coefficients,
        model_metadata["heads"], **optional,
    )


def _new_statistics(outputs: int, width: int) -> _Statistics:
    return _Statistics(np.zeros((outputs, width, width)), np.zeros((width, outputs)))


def _accumulate(state: _Statistics, z: np.ndarray, target: np.ndarray,
    weights: np.ndarray, bypass: np.ndarray) -> None:
    state.windows += len(z)
    for output in range(target.shape[1]):
        active = ~bypass[:, output]
        state.fitted_cells += int(active.sum())
        state.bypassed_cells += int((~active).sum())
        if active.any():
            za, qa = z[active], weights[active, output]
            state.gram[output] += za.T @ (qa[:, None] * za)
            state.rhs[:, output] += za.T @ (qa * target[active, output])


def prepare_ridge(batches: Iterable[WindowBatch], config: RidgeConfig) -> RidgeProblem:
    statistics = {}
    joint_features, joint_targets = [], []
    joint_panel_id = None
    L = H = None
    fitted_members = None
    for batch in batches:
        if not len(batch.x):
            continue
        if not np.isfinite(batch.x).all() or not np.isfinite(batch.y).all():
            raise ValueError("Ridge fitting requires finite inputs and targets")
        B, N, batch_L = batch.x.shape
        batch_H = batch.y.shape[2]
        if L is None:
            L, H, fitted_members = batch_L, batch_H, tuple(batch.member_ids)
        elif (L, H, fitted_members) != (batch_L, batch_H, tuple(batch.member_ids)):
            raise ValueError("Training batches must share L, H, and members")
        features, offset, scale = prepared_channels(batch.x, config)
        target = (batch.y - offset) / scale
        weights = np.square(scale[:, :, 0] / loss_denominator(batch.x, config.loss, config.epsilon))
        bypass_members = batch.constant if config.repeat_constant else np.zeros_like(batch.constant)
        if config.mode == "joint":
            if (config.loss, config.normalization, config.repeat_constant) == ("mse", "none", False):
                joint_features.append(features.reshape(B, -1))
                joint_targets.append(target.reshape(B, N * H))
                if joint_panel_id not in {None, batch.panel_id}:
                    raise ValueError("One joint problem cannot mix semantic panels")
                joint_panel_id = batch.panel_id
                continue
            z = _with_intercept(features.reshape(B, -1), config.intercept)
            y = target.reshape(B, N * H)
            q = np.repeat(weights, H, axis=1)
            bypass = np.repeat(bypass_members, H, axis=1)
            state = statistics.setdefault(batch.panel_id, _new_statistics(N * H, z.shape[1]))
            _accumulate(state, z, y, q, bypass)
        elif config.mode == "shared":
            z = _with_intercept(features.reshape(B * N, -1), config.intercept)
            y = target.reshape(B * N, H)
            q = np.repeat(weights.reshape(B * N, 1), H, axis=1)
            bypass = np.repeat(bypass_members.reshape(B * N, 1), H, axis=1)
            state = statistics.setdefault("shared", _new_statistics(H, z.shape[1]))
            _accumulate(state, z, y, q, bypass)
        else:
            for index, member in enumerate(batch.member_ids):
                z = _with_intercept(features[:, index], config.intercept)
                q = np.repeat(weights[:, index:index + 1], H, axis=1)
                bypass = np.repeat(bypass_members[:, index:index + 1], H, axis=1)
                state = statistics.setdefault(str(member), _new_statistics(H, z.shape[1]))
                _accumulate(state, z, target[:, index], q, bypass)
    if joint_features:
        return RidgeProblem(config, int(L), int(H), fitted_members, {},
            np.concatenate(joint_features), np.concatenate(joint_targets), joint_panel_id)
    if not statistics:
        raise ValueError("No eligible training windows")
    return RidgeProblem(config, int(L), int(H), fitted_members, statistics)


def solve_ridge(problem: RidgeProblem, alpha: float | None = None,
    *, calculate_condition: bool = True) -> FittedRidge:
    config = problem.config if alpha is None else replace(problem.config, alpha=float(alpha))
    if problem.joint_features is not None:
        z, y = problem.joint_features, problem.joint_targets
        primal_bytes = 8 * (z.shape[1] ** 2 + z.shape[1] * y.shape[1])
        if z.shape[1] <= len(z) and primal_bytes <= 1_000_000_000:
            feature_mean = z.mean(axis=0) if config.intercept else np.zeros(z.shape[1])
            target_mean = y.mean(axis=0) if config.intercept else np.zeros(y.shape[1])
            centered_z, centered_y = z - feature_mean, y - target_mean
            system = centered_z.T @ centered_z / len(z) + config.alpha * np.eye(z.shape[1])
            rhs = centered_z.T @ centered_y / len(z)
            slopes = (np.linalg.lstsq(system, rhs, rcond=config.solver_rcond)[0]
                if config.alpha == 0 else np.linalg.solve(system, rhs))
            coefficient = (np.vstack((slopes, target_mean - feature_mean @ slopes))
                if config.intercept else slopes)
            condition = float(np.linalg.cond(system)) if calculate_condition else None
            diagnostics = {problem.joint_panel_id: {"windows": len(z),
                "inputs": z.shape[1], "outputs": y.shape[1],
                "coefficient_shape": list(coefficient.shape),
                "regularized_condition_number": condition if condition is not None and np.isfinite(condition) else None,
                "normal_equation_residual": float(np.linalg.norm(system @ slopes - rhs))}}
            return FittedRidge(config, problem.L, problem.H, problem.member_ids,
                {problem.joint_panel_id: coefficient}, diagnostics)
        feature_mean = z.mean(axis=0, keepdims=True) if config.intercept else np.zeros((1, z.shape[1]))
        target_mean = y.mean(axis=0, keepdims=True) if config.intercept else np.zeros((1, y.shape[1]))
        centered_z, centered_y = z - feature_mean, y - target_mean
        system = centered_z @ centered_z.T / len(z) + config.alpha * np.eye(len(z))
        dual = (np.linalg.lstsq(system, centered_y, rcond=config.solver_rcond)[0]
            if config.alpha == 0 else np.linalg.solve(system, centered_y))
        condition = float(np.linalg.cond(system)) if calculate_condition else None
        diagnostics = {problem.joint_panel_id: {"windows": len(z),
            "inputs": z.shape[1], "outputs": y.shape[1],
            "dual_shape": list(dual.shape),
            "regularized_condition_number": condition if condition is not None and np.isfinite(condition) else None,
            "dual_equation_residual": float(np.linalg.norm(system @ dual - centered_y))}}
        return FittedRidge(config, problem.L, problem.H, problem.member_ids, {}, diagnostics,
            centered_z, dual, feature_mean, target_mean)
    coefficients, diagnostics = {}, {}
    for key, state in problem.statistics.items():
        width, outputs = state.rhs.shape
        coefficient = np.zeros((width, outputs))
        conditions, residuals = [], []
        penalty = np.full(width, config.alpha)
        if config.intercept:
            penalty[-1] = 0
        for output in range(outputs):
            system = state.gram[output] / state.windows + np.diag(penalty)
            rhs = state.rhs[:, output] / state.windows
            coefficient[:, output] = (np.linalg.lstsq(system, rhs, rcond=config.solver_rcond)[0]
                if config.alpha == 0 else np.linalg.solve(system, rhs))
            if calculate_condition:
                value = float(np.linalg.cond(system))
                conditions.append(value if np.isfinite(value) else None)
            residuals.append(float(np.linalg.norm(system @ coefficient[:, output] - rhs)))
        coefficients[key] = coefficient
        diagnostics[key] = {"windows": state.windows, "outputs": outputs,
            "fitted_cells": state.fitted_cells, "bypassed_cells": state.bypassed_cells,
            "regularized_condition_numbers": conditions,
            "coefficient_norm": float(np.linalg.norm(coefficient[:-1] if config.intercept else coefficient)),
            "max_normal_equation_residual": max(residuals)}
    return FittedRidge(config, problem.L, problem.H, problem.member_ids, coefficients, diagnostics)


def fit_ridge(batches: Iterable[WindowBatch], config: RidgeConfig) -> FittedRidge:
    return solve_ridge(prepare_ridge(batches, config))
