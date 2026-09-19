# Architecture

The executable path is:

```text
saved-Arrow entries + datasets.yaml semantics
  -> scalar item/channel identities
  -> one panel per physical quantity, N aligned members per panel
  -> TIME chronological multivariate windows (B,N,L) -> (B,N,H)
  -> exact ridge statistics or exact dual state
  -> validation-selected alpha
  -> ridge, persistence and full-prefix Seasonal Naive evaluation
  -> task manifests, study reports and paired figures
```

`data/time.py` interprets the two semantic axes. A temperature/load dataset
becomes two logical panels. A collection of load sensors becomes one panel.
Mixed datasets, such as water-quality measurements at several stations, become
one panel per measured quantity with stations retained as members. The loader
requires members of a joint panel to share start, frequency and length.

`data/windows.py` preserves the panel shape. Missing histories follow TIME
forward filling. Training requires complete finite N×H labels; evaluation may
retain partial labels. MASE scales use all finite seasonal pairs preceding each
origin. Seasonal Naive observes the complete prefix independently of L.
Effective L is `min(proposed_L, T-test_length)` for each panel, where the second
term is the history available at the earliest test origin. When validation
starts earlier than L, alpha selection starts at the first validation origin
with a complete context; the TIME validation boundary and labels do not move.

`proposal/ridge.py` owns all three linear modes. Joint raw-MSE fitting chooses
between exact primal and exact dual ridge according to matrix shape and a
one-gigabyte primal-state bound. Both solve the same objective. Independent
mode owns one head per member. Shared mode pools member windows into one head,
which enables unseen-member evaluation. Normalization and DLinear-style
moving-average decomposition operate per member before mode-specific design.

`pipeline/tasks.py` allocates exact task manifests, fits training-only
statistics, selects alpha on validation, evaluates populations and writes
artifacts. `pipeline/default.py` composes the three study families.
`pipeline/report.py` loads the exact completed tasks named by one launch
manifest; only the user-generalization family aggregates seeds. `results/`
owns metrics, W10 user tails, Seasonal ratios, tables and reductions.
`visualization/` owns generic TIME plots, L-H heatmaps and seed error bars.

The three root Slurm files are the only submission fronts. They source
`src/slurm/study.sh`, which composes fit/evaluate and reporting inside one
allocation. Runtime paths, recovery status, resource snapshots, transfer and
publication remain project-scoped.
