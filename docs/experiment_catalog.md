# Experiment catalog

## Semantic dataset contract

Each of the 50 TIME dataset-frequency entries declares `linear_time.structure`
and `quantity_grouping` in `datasets.yaml`. Current declarations contain 3
univariate, 12 repeated-series, 16 distinct-quantity and 19 mixed entries.
These labels describe physical meaning rather than Arrow dimensionality:
sensors stored as channels are repeated series, while unrelated scalar items
may be distinct quantities.

Distinct quantities are separate logical datasets in every experiment.
Repeated entities remain members of one panel. Special grouping rules split
regional demand/price by channel prefix and supply-chain quantity by suffix.
The task manifest records the structure, panel identity and member count.

## 01 — default

Execution front: `01_default.slurm`.

The default has no seed and no user split. It excludes the four long TIME
entries already excluded by the curated benchmark and runs every remaining
dataset-frequency entry over all nine declared L-H cells. Every physical
quantity panel gets a joint N·L→N·H model. Thus the final fit count is obtained
from the semantic panels present in the prepared Arrow snapshot; it is not the
old 90-native-task count and is not multiplied by seeds.

The grid targets calendar-scale contexts and horizons. Hourly entries use
24/168/720 for day/week/month when TIME intervals allow it. H is shortened
when the fixed validation or test interval is smaller. At runtime L is capped
at the history available at the earliest test date. Proposed term/value and
effective value both remain part of task identity. Alpha selection uses the
first validation origin with the effective history when validation starts
earlier. Reports are launch-exact and reject missing or incomplete tasks.

## 02 — user generalization

Execution front: `02_user_generalization.slurm`.

This experiment includes only `multiple_series` and `mixed` declarations and
only panels with at least two members. Seeds 0, 1 and 2 independently permute
members into 80% seen and 20% unseen subsets. The single shared independent
L→H ridge is trained on seen members. Seen and unseen validation/test results
are reported separately. Arithmetic seed means, sample SD (`ddof=1`) and mean
± SD bounds quantify split sensitivity.

## 03 — variate modes

Execution front: `03_variate_modes.slurm`.

This experiment includes every semantic panel with at least two members and
uses all members without seeding. It compares independent per-member heads,
the jointly informed N·L→N·H head, and one shared independently applied head.
Every mode predicts every member separately; the distinction is which histories
inform a forecast and which coefficients are shared.

## Common factors and evaluation

All studies support MSE/NMSE/relative-MSE fitting loss, none/center/z-score/
relative-mean normalization, optional moving-average detrending, constant
window filtering, repeat-constant outputs, and five validation-selected alpha
values. Validation never enters coefficient fitting.

Report MASE, MAE, MSE, NMSE and relative MSE plus W10 of each. W10 averages
the worst `ceil(10% of finite members)` member means. MASE has no epsilon and
zero seasonal scale is undefined. NMSE and relative MSE have additive epsilon.
Scaled MASE divides by matched Seasonal Naive task means. Within-task
population dispersion and across-seed user-split dispersion remain separate.
