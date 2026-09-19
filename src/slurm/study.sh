#!/bin/bash
set -euo pipefail

STUDY="${LINEAR_STUDY:?LINEAR_STUDY must be set by the Slurm front}"
EXPERIMENT_MODE="${EXPERIMENT_MODE:-full}"
STAGES="${STAGES:-fit,report}"
case "$STUDY" in default|user_generalization|variate_modes) ;; *) exit 2 ;; esac
case "$EXPERIMENT_MODE" in test|full) ;; *) exit 2 ;; esac
case "$STAGES" in fit,report|fit|report) ;; *) exit 2 ;; esac

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export TIME_WORKFLOW_NAME="$STUDY" TIME_TASK_NAME=linear TIME_STATUS_NAME=linear
export TIME_RESULT_SCOPE="$TIME_OUTPUTS/$STUDY/tasks"
source "$PROJECT_ROOT/src/slurm/workflow_common.sh"
time_workflow_init
time_log "study=$STUDY mode=$EXPERIMENT_MODE stages=$STAGES scientific_device=cpu dtype=float64"

case "$STUDY" in
    default) run_module=scripts.run_default ;;
    user_generalization) run_module=scripts.run_user_generalization ;;
    variate_modes) run_module=scripts.run_variate_modes ;;
esac
if [[ ",$STAGES," == *,fit,* ]]; then
    time_stage_start fit_evaluate
    srun --ntasks=1 uv run --no-sync python -m "$run_module" "experiment_mode=$EXPERIMENT_MODE"
    time_stage_complete
fi
if [[ ",$STAGES," == *,report,* ]]; then
    time_stage_start report
    srun --ntasks=1 uv run --no-sync python -m scripts.report_study \
        "study=$STUDY" "experiment_mode=$EXPERIMENT_MODE"
    time_stage_complete
fi
time_workflow_complete
