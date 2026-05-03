#!/usr/bin/env bash
#SBATCH --job-name=qtqp_clarabel_lp_sweep
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
#SBATCH --partition=savio4_htc
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=64G
#SBATCH --account=fc_radar
#
# Run the QTQP + Clarabel LP sweep on Berkeley Savio's `savio4_htc`
# partition. Pairs with configs/qtqp_and_clarabel_lp_sweep_savio.yaml,
# which pins `parallelism: 56` to match a full savio4_htc node (Intel
# Xeon Gold 6330, 56 cores).
#
# Submission (fresh run):
#     mkdir -p slurm
#     sbatch scripts/slurm_qtqp_and_clarabel_lp_sweep.sh
#
# Submission (resume an existing run dir, e.g. after walltime expired):
#     RESUME_RUN_DIR=results/qtqp_and_clarabel_lp_sweep_savio_2026-... \
#         sbatch scripts/slurm_qtqp_and_clarabel_lp_sweep.sh
#
# Notes:
#   - The YAML has `auto_prepare_data: true`. On first invocation the
#     runner downloads (a) the miplib subset from miplib.zib.de and
#     (b) the maros_meszaros_v2 zip from a private Cloudflare R2
#     bucket. Savio compute nodes have outbound network access, so
#     both work directly. To avoid downloading inside the job, run
#     `uv run bench data prepare miplib` and `uv run bench data prepare
#     maros_meszaros_v2` on a login node first.
#   - The R2 download requires credentials. Export R2_ACCESS_KEY_ID
#     and R2_SECRET_ACCESS_KEY (or AWS_ACCESS_KEY_ID /
#     AWS_SECRET_ACCESS_KEY) before `sbatch`; sbatch propagates the
#     submission environment to the job by default. The script aborts
#     early if neither pair is set.
#   - The qtqp source lives at ssh://git@github.com/google-deepmind/
#     qtqp.git on branch `berkeley-slurm`. The compute node needs an
#     SSH key with read access to that repo (run `ssh -T git@github.com`
#     once on a login node to seed known_hosts). To avoid an SSH dance
#     per job, run `uv sync --frozen --extra qtqp --extra r2` once on
#     a login node so uv.lock pins the resolved git SHA and the source
#     is cached locally.
#   - The bench CLI is managed by `uv`; subprocess workers inherit the
#     same environment so you don't need to activate a venv manually.

set -euo pipefail

# --- Paths ------------------------------------------------------------------
# Resolve repo root from this script's own location so the job runs the same
# code regardless of where sbatch was invoked.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"
mkdir -p slurm

CONFIG="configs/qtqp_and_clarabel_lp_sweep_savio.yaml"
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 1; }

# --- Cluster environment ----------------------------------------------------
# Savio doesn't ship `uv` system-wide; the recommended path is a per-user
# install at ~/.local/bin/uv (or load it from a personal module). Edit the
# next few lines for your setup.
# module load python/3.11
command -v uv >/dev/null || {
    echo "uv not found on PATH; install it (https://docs.astral.sh/uv/) " \
         "or load it via your cluster's modules." >&2
    exit 1
}

# Sync the locked deps into the project venv so worker subprocesses see the
# same environment as the main process. --frozen refuses to modify uv.lock.
#   --extra qtqp pulls in the qtqp solver (now sourced from
#       ssh://git@github.com/google-deepmind/qtqp.git@berkeley-slurm; SSH
#       access to that repo must be set up on the compute node).
#   --extra r2 pulls in boto3 so prepare_data() can fetch the
#       maros_meszaros_v2 zip from the private Cloudflare R2 bucket.
#       Requires R2_ACCESS_KEY_ID + R2_SECRET_ACCESS_KEY in env (or the
#       equivalent AWS_* names).
if [[ -z "${R2_ACCESS_KEY_ID:-}" || -z "${R2_SECRET_ACCESS_KEY:-}" ]] \
   && [[ -z "${AWS_ACCESS_KEY_ID:-}" || -z "${AWS_SECRET_ACCESS_KEY:-}" ]]; then
    echo "ERROR: R2 credentials not set. Export R2_ACCESS_KEY_ID and " \
         "R2_SECRET_ACCESS_KEY (or AWS_*) before submitting." >&2
    exit 1
fi
uv sync --frozen --extra qtqp --extra r2

# --- Header for the slurm log -----------------------------------------------
echo "==> $(date -Is) starting job"
echo "    SLURM_JOB_ID=${SLURM_JOB_ID:-(local)}"
echo "    host=$(hostname)"
echo "    partition=${SLURM_JOB_PARTITION:-?}"
echo "    repo=$REPO_ROOT"
echo "    config=$CONFIG"
echo "    cpus_per_task=${SLURM_CPUS_PER_TASK:-?}"
echo "    mem_per_node_mb=${SLURM_MEM_PER_NODE:-?}"
echo "    walltime_limit_min=${SLURM_JOB_TIME_LIMIT:-?}"
if [[ -n "${RESUME_RUN_DIR:-}" ]]; then
    [[ -d "$RESUME_RUN_DIR" ]] || {
        echo "RESUME_RUN_DIR=$RESUME_RUN_DIR does not exist" >&2; exit 1;
    }
    echo "    resuming run_dir=$RESUME_RUN_DIR"
fi
echo

# Sanity check: the config's parallelism should match what SLURM gave us.
# A mismatch is just a warning — the run will still work, just under-
# or over-subscribed.
config_parallelism=$(grep -E "^[[:space:]]*parallelism:" "$CONFIG" \
    | head -1 | awk '{print $2}')
if [[ -n "${SLURM_CPUS_PER_TASK:-}" \
      && "$config_parallelism" != "$SLURM_CPUS_PER_TASK" ]]; then
    echo "WARNING: $CONFIG has parallelism=$config_parallelism but " \
         "SLURM allocated $SLURM_CPUS_PER_TASK cpus_per_task. Edit one " \
         "to match the other." >&2
fi

# --- Run --------------------------------------------------------------------
# --no-stream-output keeps solver chatter out of the slurm-*.out file
# (per-solve stdout/stderr still land under <run_dir>/problems/.../).
# --environment-id stamps each result with the slurm job id so a future
# `bench env compare` can tell apart re-runs on different nodes.
RUN_ARGS=(
    run "$CONFIG"
    --no-stream-output
    --environment-id "slurm-${SLURM_JOB_ID:-local}"
)
if [[ -n "${RESUME_RUN_DIR:-}" ]]; then
    RUN_ARGS+=(--run-dir "$RESUME_RUN_DIR")
fi

# `bench run` echoes the run_dir on the last line of stdout. Capture it via
# tee so the slurm log shows the full live output and we still get the path
# at the end for the resume hint below.
RUN_LOG="$REPO_ROOT/slurm/run-${SLURM_JOB_ID:-local}.log"
uv run --frozen bench "${RUN_ARGS[@]}" 2>&1 | tee "$RUN_LOG"
RUN_DIR=$(tail -n 1 "$RUN_LOG")

echo
echo "==> $(date -Is) job finished"
echo "    run_dir=$RUN_DIR"
echo
echo "If walltime ran out before the sweep completed, resubmit with:"
echo "    RESUME_RUN_DIR=$RUN_DIR sbatch $0"
