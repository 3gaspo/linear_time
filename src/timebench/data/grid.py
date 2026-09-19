"""Dataset-specific Cartesian product of three L and three H lengths."""

from dataclasses import dataclass
from pathlib import Path

TERMS = ("short", "medium", "long")
DEFAULT_EXCLUDED = ("Coastal_T_S/5T", "current_velocity/20T", "azure2019_D/5T", "azure2019_I/5T")
CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config"


@dataclass(frozen=True)
class WindowSetting:
    dataset: str
    L_term: str
    H_term: str
    L: int
    H: int
    val_length: int
    test_length: int
    term: str = ""


def load_grid(catalog_path: str | Path | None = None, grid_path: str | Path | None = None) -> dict[str, list[WindowSetting]]:
    import yaml

    catalog = yaml.safe_load(Path(catalog_path or CONFIG_ROOT / "datasets.yaml").read_text(encoding="utf-8-sig"))["datasets"]
    lengths = yaml.safe_load(Path(grid_path or CONFIG_ROOT / "ridge_grid.yaml").read_text(encoding="utf-8-sig"))["datasets"]
    if set(catalog) != set(lengths):
        raise ValueError("Ridge lengths must cover exactly the TIME dataset catalog")
    grid = {}
    for dataset, entry in lengths.items():
        reference = catalog[dataset]
        Ls, Hs = entry["context_lengths"], entry["horizon_lengths"]
        for axis in (Ls, Hs):
            if set(axis) != set(TERMS) or not 0 < axis["short"] < axis["medium"] < axis["long"]:
                raise ValueError(f"{dataset} needs three increasing positive lengths per axis")
        minimum_interval = min(reference["val_length"], reference["test_length"]) if reference["val_length"] else reference["test_length"]
        if Hs["long"] > minimum_interval:
            raise ValueError(f"{dataset}: H exceeds a TIME evaluation interval")
        grid[dataset] = [WindowSetting(
            dataset, lt, ht, int(Ls[lt]), int(Hs[ht]),
            int(reference["val_length"]), int(reference["test_length"]),
        ) for lt in TERMS for ht in TERMS]
    return grid


def load_benchmark_tasks(L_term: str = "short", excluded_datasets=DEFAULT_EXCLUDED) -> list[WindowSetting]:
    """The curated TIME catalog's 90 native dataset/frequency/term tasks.

    Additional child horizons and the full nine-cell L-H study do not expand
    this default benchmark. Context length is selected independently of H.
    """
    import yaml

    if L_term not in TERMS:
        raise ValueError("L_term must be short, medium, or long")
    catalog = yaml.safe_load((CONFIG_ROOT / "datasets.yaml").read_text(encoding="utf-8-sig"))["datasets"]
    grid = load_grid()
    tasks = []
    for dataset, reference in catalog.items():
        if dataset in excluded_datasets:
            continue
        for term in TERMS:
            if term not in reference:
                continue
            H = int(reference[term]["prediction_length"])
            # Native benchmark horizons and the calendar-scale L-H study are
            # independent. A native H need not occupy one of the three study
            # cells; only the selected dataset-specific L is shared.
            context = next(s for s in grid[dataset] if s.L_term == L_term)
            tasks.append(WindowSetting(dataset, L_term, term, context.L, H,
                context.val_length, context.test_length, term))
    if len(tasks) != 90:
        raise ValueError("The default TIME benchmark must contain exactly 90 native tasks")
    return tasks
