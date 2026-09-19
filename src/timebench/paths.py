"""Project-scoped subset of Improved TIME's environment path contract."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _configured_path(variable: str, fallback: Path) -> Path:
    value = os.environ.get(variable)
    return Path(value).expanduser() if value else fallback


def dataset_storage_root() -> Path:
    data_root = _configured_path("TIME_DATA_ROOT", PROJECT_ROOT / "datasets")
    return _configured_path("TIME_DATASET", data_root / "hf_dataset")


def outputs_root() -> Path:
    return _configured_path("TIME_OUTPUTS", PROJECT_ROOT / "outputs")


def logs_root() -> Path:
    return _configured_path("TIME_LOGS", PROJECT_ROOT / "logs")
