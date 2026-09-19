"""Validate public views and optionally render Linear TIME LaTeX documents."""

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VIEWS = [ROOT / "README.md", ROOT / "docs/architecture.md",
    ROOT / "docs/experiment_catalog.md", ROOT / "docs/results_recap.md",
    ROOT / "latex/method_overview.tex", ROOT / "latex/experiment_guideline.tex",
    ROOT / "latex/executive_summary.tex"]
FRONTS = ["01_default.slurm", "02_user_generalization.slurm", "03_variate_modes.slurm"]


def validate() -> None:
    missing = [str(path.relative_to(ROOT)) for path in VIEWS if not path.is_file()]
    if missing:
        raise ValueError(f"Missing public views: {missing}")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    headings = [line for line in readme.splitlines() if line.startswith("## ")]
    expected = ["## Documentation map", "## Setup", "## Main executions",
        "## Outputs and cluster operations", "## Documentation maintenance"]
    if headings != expected or len(readme.splitlines()) > 160:
        raise ValueError("README heading/length contract is not satisfied")
    catalog = (ROOT / "docs/experiment_catalog.md").read_text(encoding="utf-8")
    for front in FRONTS:
        if not (ROOT / front).is_file() or front not in catalog:
            raise ValueError(f"Missing documented Slurm front: {front}")


def render(name: str) -> None:
    executable = shutil.which("pdflatex")
    if executable is None:
        raise RuntimeError("pdflatex is unavailable")
    source = ROOT / "latex" / f"{name}.tex"
    for _ in range(2):
        subprocess.run([executable, "-interaction=nonstopmode", "-halt-on-error", source.name],
            cwd=source.parent, check=True)
    for suffix in (".aux", ".log", ".out"):
        source.with_suffix(suffix).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", choices=("method", "guideline", "summary", "all"))
    args = parser.parse_args()
    validate()
    selected = {"method": ["method_overview"], "guideline": ["experiment_guideline"],
        "summary": ["executive_summary"], "all": ["method_overview", "experiment_guideline", "executive_summary"]}
    for name in selected.get(args.render, []):
        render(name)


if __name__ == "__main__":
    main()
