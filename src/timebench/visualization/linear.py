"""Generic two-parameter, L-H, and within-task dispersion plots."""

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def save_figure(figure, destination: str | Path) -> list[Path]:
    """Return both exports for inclusion in the owning report manifest."""
    destination = Path(destination)
    if destination.suffix in {".png", ".pdf"}:
        destination = destination.with_suffix("")
    destination.parent.mkdir(parents=True, exist_ok=True)
    paths = [Path(str(destination) + suffix) for suffix in (".png", ".pdf")]
    for path in paths:
        figure.savefig(path, bbox_inches="tight", dpi=180)
    return paths


def plot_parameter_heatmap(
    records: Sequence[Mapping], *, x: str, y: str, metric: str,
    destination: str | Path, filters: Mapping | None = None,
    title: str | None = None,
) -> list[Path]:
    """One selected value per parameter cell; never average duplicate runs.

    Records are flat mappings with x/y fields and metric scalar values. The
    caller selects the dataset, population, model and repeat/config policies.
    Missing cells remain visible; all L-H comparisons can therefore share axes.
    """
    import matplotlib.pyplot as plt

    selected = [r for r in records if all(r.get(key) == value for key, value in (filters or {}).items())]
    if not selected:
        raise ValueError("No plot records match the filters")
    xs = sorted({r[x] for r in selected})
    ys = sorted({r[y] for r in selected})
    values = np.full((len(ys), len(xs)), np.nan)
    occupied = set()
    for record in selected:
        position = (ys.index(record[y]), xs.index(record[x]))
        if position in occupied:
            raise ValueError("Duplicate heatmap cell: select a single model/configuration/repeat first")
        occupied.add(position)
        values[position] = record.get(metric) if record.get(metric) is not None else np.nan
    figure, axis = plt.subplots(figsize=(max(5, len(xs)), max(4, len(ys) * 0.7)))
    mesh = axis.imshow(np.ma.masked_invalid(values), origin="lower", aspect="auto", cmap="viridis")
    axis.set(xticks=range(len(xs)), xticklabels=xs, yticks=range(len(ys)), yticklabels=ys,
        xlabel=x, ylabel=y, title=title or metric)
    for (row, column), value in np.ndenumerate(values):
        axis.text(column, row, f"{value:.4g}" if np.isfinite(value) else "missing", ha="center", va="center",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2})
    figure.colorbar(mesh, ax=axis, label=metric)
    paths = save_figure(figure, destination)
    plt.close(figure)
    return paths


def plot_lh_heatmap(records: Sequence[Mapping], *, metric: str, destination: str | Path,
    filters: Mapping | None = None, title: str | None = None) -> list[Path]:
    return plot_parameter_heatmap(records, x="L", y="H", metric=metric,
        destination=destination, filters=filters, title=title)


def plot_mean_std(records: Sequence[Mapping], *, metric: str, destination: str | Path,
    model_field: str = "model") -> list[Path]:
    """One point per selected model/task, using within-task population std.

    The metric_std field must come from metric cells, never dispersion across
    seeds. Configuration/repeat aggregation belongs to the report selector.
    """
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 5))
    for model in sorted({r[model_field] for r in records}):
        selected = [r for r in records if r[model_field] == model
            and r.get(metric) is not None and r.get(f"{metric}_std") is not None]
        axis.scatter([r[metric] for r in selected], [r[f"{metric}_std"] for r in selected], label=model, alpha=0.75)
    axis.set(xlabel=f"Mean {metric}", ylabel=f"Population standard deviation of {metric}")
    axis.legend()
    paths = save_figure(figure, destination)
    plt.close(figure)
    return paths


def plot_seed_intervals(records: Sequence[Mapping], *, metric: str, destination: str | Path) -> list[Path]:
    """H on x, seed mean +/- sample SD on y, one curve per L."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 4))
    for L in sorted({r["L"] for r in records}):
        selected = sorted([r for r in records if r["L"] == L and r.get(metric) is not None
            and r.get(f"{metric}_seed_std") is not None], key=lambda r: r["H"])
        axis.errorbar([r["H"] for r in selected], [r[metric] for r in selected],
            yerr=[r[f"{metric}_seed_std"] for r in selected], marker="o", capsize=4, label=f"L={L}")
    axis.set(xlabel="H", ylabel=f"{metric}: seed mean +/- SD")
    axis.legend()
    paths = save_figure(figure, destination)
    plt.close(figure)
    return paths
