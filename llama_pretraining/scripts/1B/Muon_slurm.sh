#!/bin/bash
#SBATCH --job-name=muon-1b
#SBATCH --account=le-lab
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:4
#SBATCH --mem=100GB
#SBATCH --time=336:00:00
#SBATCH --output=muon-1b-%j.out
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tucnguye@iu.edu

set -euo pipefail

REPO=/data/project/le-lab/Soren/llama_pretraining
RUN_ROOT=/data/project/le-lab/Soren/runs

# Plain Muon baseline for the 1B C4 runs. Steps/warmup/eval match
# HTMuonInterval_slurm.sh so the curves are directly comparable; Muon uses the
# same lr/lrmuon but takes no --power / --interval (it always runs the 5-step
# Newton-Schulz orthogonalization, never the SVD path).
#
# Rank count follows the allocation; override with GPUS=<n> without touching
# --gres if you want fewer ranks than the allocation.
GPUS=${GPUS:-${SLURM_GPUS_ON_NODE:-4}}
MICRO_BATCH=${MICRO_BATCH:-32}        # per-GPU; 128 OOMs a 1B model even on 48GB
TOTAL_BATCH=${TOTAL_BATCH:-512}       # global batch, held fixed across GPU counts
RUN_NAME=${RUN_NAME:-muon_1b}

if (( TOTAL_BATCH % (MICRO_BATCH * GPUS) != 0 )); then
    echo "TOTAL_BATCH ($TOTAL_BATCH) must be divisible by MICRO_BATCH*GPUS ($((MICRO_BATCH * GPUS)))" >&2
    exit 1
fi

# --- environment ------------------------------------------------------------
nvidia-smi

# conda's activate.d hooks (gcc_linux-64) reference unset vars, which is fatal
# under `set -u` -- relax nounset just for activation, then restore it.
set +u
eval "$(conda shell.bash hook)"
conda activate /data/project/le-lab/conda_env/htmuon
set -u

cd "$REPO" # configs/ and the entrypoint are relative

# Keep HF and W&B caches off the home quota.
export HF_HOME=${HF_HOME:-/data/project/le-lab/conda_env/.hf_cache}
# HF_HOME relocates the token lookup to $HF_HOME/token, so the login token in
# ~/.cache/huggingface is invisible and every C4 request goes out anonymous --
# anonymous traffic gets rate-limited (HTTP 429) on long runs. Pass it through.
export HF_TOKEN=${HF_TOKEN:-$(cat /u/tucnguye/.cache/huggingface/token 2>/dev/null || true)}
export WANDB_DIR=${WANDB_DIR:-$RUN_ROOT}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

export WANDB_API_KEY=${WANDB_API_KEY:?set it in your shell}
export WANDB_PROJECT=soren
export WANDB_MODE=online


SAVE_DIR=$RUN_ROOT/${RUN_NAME}_${SLURM_JOB_ID:-local}
mkdir -p "$SAVE_DIR"

# Unique port so concurrent jobs on the same node do not collide.
MASTER_PORT=$(( 20000 + (${SLURM_JOB_ID:-$RANDOM} % 10000) ))

# Resume: explicit RESUME_FROM wins; otherwise pick up the newest checkpoint in
# SAVE_DIR. Slurm keeps the same job id across --requeue, so SAVE_DIR is stable
# and a requeued job continues where it left off instead of restarting at step 0.
RESUME_ARGS=()
if [[ -z "${RESUME_FROM:-}" ]]; then
    LATEST=$(ls -d "$SAVE_DIR"/model_* 2>/dev/null | sed 's#.*/model_##' | sort -n | tail -1 || true)
    [[ -n "$LATEST" ]] && RESUME_FROM="$SAVE_DIR/model_$LATEST"
fi
if [[ -n "${RESUME_FROM:-}" ]]; then
    echo "resuming from $RESUME_FROM"
    RESUME_ARGS=(--continue_from "$RESUME_FROM")
fi

echo "gpus=$GPUS micro_batch=$MICRO_BATCH total_batch=$TOTAL_BATCH" \
     "grad_accum=$((TOTAL_BATCH / (MICRO_BATCH * GPUS))) save_dir=$SAVE_DIR"

torchrun --nproc_per_node="$GPUS" --master_port="$MASTER_PORT" --master_addr=localhost \
    torchrun_main_HTMuon.py \
    --model_config configs/llama_1b.json \
    --optimizer muon \
    --seed 5 \
    --lr 0.001 \
    --lrmuon 5e-3 \
    --batch_size "$MICRO_BATCH" \
    --total_batch_size "$TOTAL_BATCH" \
    --num_training_steps 15000 \
    --warmup_steps 1500 \
    --weight_decay 0.1 \
    --dtype bfloat16 \
    --eval_every 150 \
    --wandb_name "$RUN_NAME" \
    --target_eval_tokens 10_000_000 \
    --save_every 1000 \
    --save_dir "$SAVE_DIR" \
    --workers 4 \
    "${RESUME_ARGS[@]}"  