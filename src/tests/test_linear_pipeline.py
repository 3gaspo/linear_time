"""Focused numerical, semantic, lifecycle, and scheduler checkpoint."""

import ast
import json
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np
import yaml

from timebench.data.grid import WindowSetting, load_grid
from timebench.data.splits import split_users
from timebench.data.time import TimePanel, load_dataset_semantics
from timebench.data.windows import WindowBatch, iter_window_batches, maximum_test_context
from timebench.pipeline import tasks
from timebench.pipeline.runs import allocate_run, load_manifest
from timebench.proposal.ridge import RidgeConfig, fit_ridge, prepare_ridge
from timebench.results.evaluate import evaluate_batches
from timebench.results.metrics import METRICS, compute_window_metrics, summarize_metrics
from timebench.results.reporting import aggregate_seed_rows, seed_statistics
from timebench.visualization.linear import plot_lh_heatmap

ROOT = Path(__file__).resolve().parents[2]


def panel(members=3, length=80):
    time = np.arange(length, dtype=float)
    values = np.stack([.1 * time + np.sin(time / (i + 2)) + i for i in range(members)])
    return TimePanel("load", "load", tuple(f"u{i}" for i in range(members)),
        values, "H", 4, "2000", "multiple_series")


def batch(x, y, ids=None, panel_id="load"):
    x, y = np.asarray(x, float), np.asarray(y, float)
    B, N, _ = x.shape
    return WindowBatch(x, y, np.asarray(ids or [f"u{i}" for i in range(N)]), panel_id,
        np.arange(B), np.ones((B, N)), x.std(axis=2) < 1e-8)


def config(output):
    value = yaml.safe_load((ROOT / "src/conf/config.yaml").read_text())
    value["model"]["epsilon"] = value["evaluation"]["epsilon"] = value["epsilon"]
    value["validation"]["metric"] = value["model"]["loss"]
    value["artifacts"]["output_root"] = str(output)
    value["report"]["plots"] = False
    return value


class LinearPipelineCheck(unittest.TestCase):
    def test_calendar_grid_and_semantics(self):
        grid = load_grid()
        self.assertEqual(len(grid), 50)
        self.assertTrue(all(len(settings) == 9 for settings in grid.values()))
        hourly = grid["SG_PM25/H"]
        self.assertEqual(sorted({s.L for s in hourly}), [24, 168, 720])
        self.assertEqual(sorted({s.H for s in hourly}), [24, 168, 720])
        catalog = yaml.safe_load((ROOT / "src/timebench/config/datasets.yaml").read_text())["datasets"]
        counts = {kind: 0 for kind in ("univariate", "multiple_series", "multiple_quantities", "mixed")}
        for name in catalog:
            counts[load_dataset_semantics(name)["structure"]] += 1
        self.assertEqual(counts, {"univariate": 3, "multiple_series": 12,
            "multiple_quantities": 16, "mixed": 19})
        self.assertEqual(load_dataset_semantics("Water_Quality_Darwin/15T")["quantity_grouping"], "variate")
        self.assertEqual(load_dataset_semantics("SG_PM25/H")["structure"], "multiple_series")

    def test_context_cap_and_panel_windows(self):
        source = panel(3, 50)
        self.assertEqual(maximum_test_context(source, 10), 40)
        batches = list(iter_window_batches(source, members=None, split="test",
            L=20, H=5, val_length=10, test_length=10))
        self.assertEqual(batches[0].x.shape, (2, 3, 20))
        self.assertEqual(batches[0].y.shape, (2, 3, 5))
        self.assertEqual(batches[0].seasonal_prediction.shape, (2, 3, 5))
        np.testing.assert_array_equal(batches[0].origins, [40, 45])
        validation = list(iter_window_batches(source, members=None, split="valid",
            L=35, H=5, val_length=10, test_length=10))
        np.testing.assert_array_equal(validation[0].origins, [35])

    def test_joint_exact_ridge_is_nl_to_nh(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(4, 2, 3))
        z = x.reshape(4, 6)
        coefficient = rng.normal(size=(6, 4))
        y = (z @ coefficient).reshape(4, 2, 2)
        fitted = fit_ridge([batch(x, y)], RidgeConfig(mode="joint", intercept=False, alpha=.01))
        self.assertEqual(fitted.joint_dual.shape, (4, 4))
        centered_z = z
        centered_y = y.reshape(4, 4)
        system = centered_z @ centered_z.T / 4 + .01 * np.eye(4)
        expected_dual = np.linalg.solve(system, centered_y)
        np.testing.assert_allclose(fitted.joint_dual, expected_dual)
        expected = (centered_z @ centered_z.T / 4) @ expected_dual
        np.testing.assert_allclose(fitted.predict(batch(x, y)), expected.reshape(4, 2, 2))

    def test_three_variate_modes(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(8, 3, 4))
        y = np.repeat(x[:, :, -1:], 2, axis=2)
        for mode, heads, width in (("joint", 1, 13), ("independent", 3, 5), ("shared", 1, 5)):
            fitted = fit_ridge([batch(x, y)], RidgeConfig(mode=mode, alpha=.1))
            self.assertEqual(len(fitted.coefficients), heads if mode != "joint" else 0)
            self.assertTrue(all(value.shape[0] == width for value in fitted.coefficients.values()))
            self.assertEqual(fitted.predict(batch(x, y)).shape, y.shape)
        shared = fit_ridge([batch(x[:, :2], y[:, :2], ["a", "b"])], RidgeConfig(mode="shared"))
        unseen = batch(x[:, 2:], y[:, 2:], ["new"])
        self.assertEqual(shared.predict(unseen).shape, (8, 1, 2))

    def test_metrics_and_seed_scope(self):
        target = np.array([[[1., np.nan], [2., 4.]]])
        prediction = np.array([[[3., 9.], [2., 2.]]])
        context = np.array([[[1., 3.], [2., 2.]]])
        metrics = compute_window_metrics(prediction, target, context, np.array([[2., 1.]]), epsilon=.1)
        self.assertEqual(metrics["mae"].shape, (1, 2))
        self.assertEqual(metrics["mase"][0, 0], 1)
        users = np.array(["a", "b"])
        summary = summarize_metrics({name: metrics[name] for name in METRICS}, users)
        self.assertEqual(summary["mse"]["finite_users"], 2)
        self.assertEqual(seed_statistics([1, 2, 3])["seed_std"], 1)
        split = split_users(["a", "b", "c", "d", "e"], .8, 0)
        self.assertNotEqual(split.seen, split_users(["a", "b", "c", "d", "e"], .8, 1).seen)

    def test_panel_task_and_seedless_default_path(self):
        setting = WindowSetting("SG_PM25/H", "short", "short", 12, 4, 12, 12)
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as directory:
            values = config(directory)
            values["experiment"] = values["study"] = "default"
            def short_allocate(identity_root, **kwargs):
                return allocate_run(Path(directory) / "default/tasks/task", **kwargs)
            with patch.object(tasks, "allocate_run", side_effect=short_allocate):
                run = tasks.run_panel_task(values, setting, panel(3))
                self.assertEqual(load_manifest(run)["status"], "completed")
                self.assertNotIn("seed_", str(run))
                manifest = load_manifest(run)
                self.assertIsNone(manifest["pipeline_config"]["seed"])
                self.assertEqual(manifest["identity"]["mode"], "joint")
                summary = json.loads((run / "metrics_summary.json").read_text())
                self.assertEqual(summary["all_test"]["seasonal_naive"]["metrics"]["scaled_mase"]["mean"], 1)

    def test_user_seed_reduction(self):
        base = {"dataset": "d", "panel": "p", "structure": "multiple_series",
            "members": 3, "L_term": "short", "H_term": "short", "L": 4, "H": 2,
            "mode": "shared", "model": "ridge", "population": "unseen_test",
            "study": "user_generalization", "selected_alpha": .1,
            "seasonal_mase_variance": 1., "inference_seconds": 1., "fit_seconds": 1.,
            "alpha_selection_seconds": 1.}
        rows = []
        for seed in (0, 1, 2):
            row = {**base, "seed": seed}
            for metric in METRICS:
                row.update({metric: seed + 1., f"w10_{metric}": seed + 1.,
                    f"{metric}_std": 1., f"{metric}_variance": 1., f"{metric}_user_mean": 1.})
            row.update(scaled_mase=seed + 1., w10_scaled_mase=seed + 1.)
            rows.append(row)
        reduced = aggregate_seed_rows(rows, [0, 1, 2])[0]
        self.assertEqual((reduced["mse"], reduced["mse_seed_std"]), (2, 1))

    def test_plots_source_and_three_slurms(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as directory:
            records = [dict(L=L, H=H, mse=float(L + H)) for L in (4, 8, 12) for H in (2, 4, 6)]
            paths = plot_lh_heatmap(records, metric="mse", destination=Path(directory) / "grid")
            self.assertEqual({path.suffix for path in paths}, {".png", ".pdf"})
        for path in (ROOT / "src").rglob("*.py"):
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        slurms = sorted(ROOT.glob("*.slurm"))
        self.assertEqual([path.name for path in slurms], ["01_default.slurm",
            "02_user_generalization.slurm", "03_variate_modes.slurm"])
        for path in slurms:
            text = path.read_text()
            for directive in ("--gres=gpu:1", "--partition=an", "--qos=an_preemptable",
                "--exclusive", "--wckey=P12CU:DATASCIENCE"):
                self.assertIn("#SBATCH " + directive, text)
        dependencies = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
        self.assertFalse(any("torch" in item or "transformers" in item for item in dependencies))


if __name__ == "__main__":
    unittest.main()
