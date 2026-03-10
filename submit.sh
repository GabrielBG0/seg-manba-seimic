#!/bin/bash
# =============================================================================
# MambaSegNet — Seismic Facies Segmentation
# Submit with: sbatch submit.sbatch
# Resume:      just resubmit the same command — train.py auto-detects last.ckpt
# =============================================================================

# ── SLURM parameters ─────────────────────────────────────────────────────────
#SBATCH --account=spfm
#SBATCH --job-name=train:mamba-seg
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=1440
#SBATCH --partition=ict-h100
#SBATCH --output=/petrobr/parceirosbr/home/gabriel.gutierrez/github/seg-manba-seimic/logs/%j_%t_log.out
#SBATCH --error=/petrobr/parceirosbr/home/gabriel.gutierrez/github/seg-manba-seimic/logs/%j_%t_log.err
#SBATCH --open-mode=append
#SBATCH --signal=USR2@120
#SBATCH --nice=0
#SBATCH --wckey=submitit

# =============================================================================
# Edit these paths to match your HPC environment
# =============================================================================
PROJECT_DIR=/petrobr/parceirosbr/home/gabriel.gutierrez/github/seg-manba-seimic
CONTAINER=$PROJECT_DIR/seg-mamba-seismic-container.sif
DATASET_DIR=$PROJECT_DIR/dataset
CHECKPOINT_DIR=$PROJECT_DIR/checkpoints
LOG_DIR=$PROJECT_DIR/logs

mkdir -p $LOG_DIR $CHECKPOINT_DIR

# =============================================================================
# Environment
# =============================================================================
export DATASET_ROOT=$DATASET_DIR
export CHECKPOINT_DIR=$CHECKPOINT_DIR
export GPUS_PER_NODE=4

# Lightning / PyTorch distributed — SLURM sets these automatically,
# but we make them explicit for Singularity which may not inherit all env vars
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export NODE_RANK=$SLURM_NODEID

# =============================================================================
# Launch — one task per GPU, srun handles rank assignment
# =============================================================================
srun \
    --unbuffered \
    --output $LOG_DIR/%j_%t_log.out \
    --error  $LOG_DIR/%j_%t_log.err \
    singularity exec --nv \
        --bind $PROJECT_DIR:/workspace \
        --bind $DATASET_DIR:/workspace/dataset \
        $CONTAINER \
        python -u /workspace/train.py
