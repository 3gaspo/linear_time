# Linear TIME

Linear TIME studies exact ridge forecasting on the TIME datasets without
foundation models. It preserves dataset semantics: different physical
quantities become separate logical panels, while repeated users or sensors of
the same quantity stay together.

## Documentation map

- [Architecture](docs/architecture.md) traces Arrow targets through fitting and reports.
- [Experiment catalog](docs/experiment_catalog.md) defines the three Slurm experiments.
- [Method overview](latex/method_overview.pdf) gives the model and metrics.
- [Experiment guideline](latex/experiment_guideline.pdf) gives the reproducibility contract.
- [Results recap](docs/results_recap.md) records inspected evidence only.

## Setup

Use Python 3.12 and `pyproject.toml` on the execution host. Put the official
saved-Arrow TIME datasets below `datasets/hf_dataset`, or set `TIME_DATASET`.
`scripts/download_time_dataset.py` prepares an immutable official snapshot.
Set the project `src/` directory on `PYTHONPATH` for direct Python commands.

Every dataset entry in `src/timebench/config/datasets.yaml` declares one of
four structures: univariate, repeated series of one quantity, multiple
distinct quantities, or both repeated series and multiple quantities. Prefix
and suffix rules describe datasets whose quantity/entity axes are encoded in
channel names. Runtime loading verifies joint-panel alignment.

## Main executions

Three Selena fronts own the executable studies:

```bash
bash scripts/submit_study.sh default
bash scripts/submit_study.sh user_generalization
bash scripts/submit_study.sh variate_modes
```

`01_default.slurm` runs every retained dataset-frequency entry over its full
3×3 L-H grid. It uses all members, has no random split or seed, and fits one
joint model per physical quantity. With N repeated entities, the model maps
N·L inputs to N·H outputs. Exact primal ridge is used when feasible; otherwise
the mathematically equivalent exact dual system avoids materializing a huge
coefficient matrix.

`02_user_generalization.slurm` selects only repeated-user/sensor panels. Seeds
0, 1, and 2 create the 80/20 seen/unseen member split. A shared L→H model is
trained on seen members and applied independently to seen and unseen members.
Seed means and sample standard deviations belong only to this experiment.

`03_variate_modes.slurm` selects panels with at least two members and compares:

- `independent`: one L→H ridge head per member;
- `joint`: one N·L→N·H model informed by all members;
- `shared`: one pooled L→H head applied independently to every member.

All studies select alpha on validation from
`[0.0001, 0.001, 0.01, 0.1, 1]`, without refitting on validation labels. Use
`EXPERIMENT_MODE=test` for the delayed one-cell smoke profile. Hydra overrides
select loss, normalization, moving-average detrending, constant rules and
runtime values through `src/conf/config.yaml`. Validation dates step backward
from the first test date at stride `H`, never exceed the number of test dates,
and require finite L-point inputs and H-point outputs. Each task records
requested, available and usable counts. If none are usable, alpha falls back to
the configured default; fitting still requires at least one valid training
input/output window.

## Outputs and cluster operations

Task paths are readable and contain dataset, L/H term and effective sizes,
physical-quantity panel, model mode, scientific configuration, optional
`seed_<seed>`, and `run_<n>`. No hashes define identity. Effective L is capped
by the history available at the earliest test origin and both proposed and
effective L are saved in the manifest. Validation starts at its first origin
with that effective history. Each report consumes exactly the task manifests
listed by its launch, so separate launches and configurations cannot mix.

Each task stores coefficients or exact dual state, alpha scores, panel/member
metadata, window metrics, summaries and independent timings. Reusable caches
under `outputs/<study>/cache/` separately own training statistics, validation
alpha selection, fitted coefficients, and each population/split evaluation.
Changing the candidate alpha list reuses unchanged training statistics; only a
changed selected alpha invalidates coefficients and their evaluations. Reports live at
`outputs/reports/<study>/<report-id>/`. They include MASE, MAE, MSE, NMSE,
relative MSE (`rmse`), their W10 versions, matched Seasonal-scaled MASE, L-H
heatmaps and paired PNG/PDF exports. Only the user-generalization report reduces
over seeds and shows mean ± sample SD; those are dispersion bounds, not
confidence intervals.

`sync_code_to_selena.sh` preserves the remote environment and excludes data,
artifacts and private records. `sync_results_to_dgx.sh` and `publish_job.sh`
share `artifact_selection.py`; lightweight mode keeps reports and compact
metadata while excluding raw arrays. Runtime artifacts and logs remain inside
this project's Selena scratch root.
Every allocation records visible accelerators, GPU/host memory, and explicit
cgroup availability before its stages. Fit/evaluation and report stages emit
the shared selected-device event with `cpu`.

## Documentation maintenance

`python src/scripts/build_docs.py` checks the public views and Slurm coverage.
`--render method`, `--render guideline`, or `--render all` builds the LaTeX
documents. Add scientific conclusions only after inspecting current-contract
TIME artifacts. No real-data experiment has been run yet.
