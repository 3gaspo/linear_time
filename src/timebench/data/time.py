"""Semantic TIME panels for joint, independent, and shared linear models."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from timebench.paths import dataset_storage_root

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config/datasets.yaml"


@dataclass(frozen=True)
class ScalarSeries:
    series_id: str
    item_id: str
    variate: str
    values: np.ndarray
    frequency: str
    seasonality: int
    start: str


@dataclass(frozen=True)
class TimePanel:
    """One physical quantity observed for one or more entities."""

    panel_id: str
    quantity: str
    member_ids: tuple[str, ...]
    values: np.ndarray
    frequency: str
    seasonality: int
    start: str
    structure: str

    @property
    def users(self) -> tuple[str, ...]:
        return self.member_ids


def load_dataset_semantics(name: str, path: str | Path | None = None) -> dict:
    import yaml

    datasets = yaml.safe_load(Path(path or CONFIG_PATH).read_text(encoding="utf-8-sig"))["datasets"]
    semantic = datasets[name].get("linear_time")
    if semantic is None:
        raise ValueError(f"Missing linear_time semantics for {name}")
    if semantic.get("structure") not in {"univariate", "multiple_series", "multiple_quantities", "mixed"}:
        raise ValueError(f"Unknown dataset structure for {name}")
    if semantic.get("quantity_grouping") not in {"single", "series", "variate", "item", "prefix", "suffix"}:
        raise ValueError(f"Unknown quantity grouping for {name}")
    return semantic


def _load_scalars(name: str, storage_path: str | Path | None = None) -> list[ScalarSeries]:
    from datasets import load_from_disk
    from gluonts.time_feature import get_seasonality

    root = Path(storage_path) if storage_path is not None else dataset_storage_root()
    dataset = load_from_disk(str(root / name))
    scalars = []
    for entry_index, entry in enumerate(dataset):
        target = np.asarray(entry["target"], dtype=np.float64)
        if target.ndim == 1:
            target = target[None, :]
        if target.ndim != 2 or np.isinf(target).any():
            raise ValueError(f"Invalid TIME target for {name}/{entry_index}")
        item_id = str(entry.get("item_id", f"item_{entry_index}"))
        names = entry.get("variate_names") or ([item_id] if len(target) == 1 else [str(i) for i in range(len(target))])
        frequency = str(entry["freq"])
        for index, values in enumerate(target):
            scalars.append(ScalarSeries(
                series_id=f"{item_id}::variate={index}", item_id=item_id,
                variate=str(names[index]), values=values, frequency=frequency,
                seasonality=int(get_seasonality(frequency)), start=str(entry["start"]),
            ))
    if len({series.series_id for series in scalars}) != len(scalars):
        raise ValueError("TIME item/channel identifiers must be unique")
    return scalars


def _quantity(series: ScalarSeries, semantic: dict) -> str:
    grouping = semantic["quantity_grouping"]
    if grouping == "single":
        return str(semantic.get("quantity", "target"))
    if grouping == "series":
        return series.series_id
    if grouping == "variate":
        return series.variate
    if grouping == "item":
        return series.item_id
    separator = semantic["separator"]
    if separator not in series.variate:
        raise ValueError(f"Expected separator {separator!r} in {series.variate!r}")
    return (series.variate.split(separator, 1)[0] if grouping == "prefix"
        else series.variate.rsplit(separator, 1)[1])


def load_time_panels(name: str, storage_path: str | Path | None = None) -> list[TimePanel]:
    """Split distinct quantities, then retain their repeated entities jointly."""
    semantic = load_dataset_semantics(name)
    groups: dict[str, list[ScalarSeries]] = {}
    for series in _load_scalars(name, storage_path):
        groups.setdefault(_quantity(series, semantic), []).append(series)
    panels = []
    for quantity, members in groups.items():
        members.sort(key=lambda item: item.series_id)
        if len({(len(item.values), item.frequency, item.start, item.seasonality) for item in members}) != 1:
            raise ValueError(f"Joint panel {name}/{quantity} has unaligned members")
        panels.append(TimePanel(
            panel_id=quantity, quantity=quantity,
            member_ids=tuple(item.series_id for item in members),
            values=np.stack([item.values for item in members]),
            frequency=members[0].frequency, seasonality=members[0].seasonality,
            start=members[0].start, structure=semantic["structure"],
        ))
    panels.sort(key=lambda panel: panel.panel_id)
    if semantic["structure"] == "univariate" and (len(panels) != 1 or len(panels[0].member_ids) != 1):
        raise ValueError(f"Declared univariate dataset {name} is not univariate")
    if semantic["structure"] == "multiple_series" and (len(panels) != 1 or len(panels[0].member_ids) < 2):
        raise ValueError(f"Declared multiple-series dataset {name} has no repeated entities")
    return panels
