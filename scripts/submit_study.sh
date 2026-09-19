#!/bin/bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
study="${1:-default}"
case "$study" in
    default|user_generalization|variate_modes)
        NNI_FILE="${TIME_NNI_FILE:-$HOME/codes/.secrets/nni}"
        nni="$(sed -n '1p' "$NNI_FILE" | tr -d '[:space:]')"
        nni="${nni,,}"
        [[ "$nni" =~ ^[a-z][a-z0-9_-]*$ ]] || { echo "Invalid NNI" >&2; exit 2; }
        mkdir -p "/scratch/users/$nni/codes/linear_time/logs"
        case "$study" in
            default) front=01_default.slurm ;;
            user_generalization) front=02_user_generalization.slurm ;;
            variate_modes) front=03_variate_modes.slurm ;;
        esac
        ;;
    *) echo "Study must be default, user_generalization, or variate_modes" >&2; exit 2 ;;
esac
options=()
if [ -n "${SBATCH_DEPENDENCY:-}" ]; then options+=("--dependency=$SBATCH_DEPENDENCY"); fi
sbatch "${options[@]}" "$front"
