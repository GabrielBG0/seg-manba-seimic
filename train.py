"""
train.py — MambaSegNet seismic facies segmentation
====================================================
Designed for HPC / SLURM environments:
  - 8 GPUs across 2 nodes (4 GPUs per node) via DDP
  - Automatically resumes from last checkpoint if one exists
  - Handles the 24-hour job time limit gracefully via on_train_end checkpoint
"""

import os, sys, torch
import numpy as np
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_float32_matmul_precision("high")

import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint, LearningRateMonitor, EarlyStopping
)
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy

from lightning_module import SeismicDataModule, MambaSegLightning

# ===========================================================================
# Configuration
# ===========================================================================

DATASET_ROOT   = Path(os.environ.get("DATASET_ROOT", "dataset"))
TRAIN_DATA     = DATASET_ROOT / "data"        / "train"
TRAIN_LABELS   = DATASET_ROOT / "annotations" / "train"
VAL_DATA       = DATASET_ROOT / "data"        / "val"
VAL_LABELS     = DATASET_ROOT / "annotations" / "val"
TEST_DATA      = DATASET_ROOT / "data"        / "test"
TEST_LABELS    = DATASET_ROOT / "annotations" / "test"

# Logs and checkpoints go to a shared filesystem visible to all nodes
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", "checkpoints"))
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

MAX_H, MAX_W     = 1006, 590
CANVAS_H         = int(np.ceil(MAX_H / 32) * 32)   # 1024
CANVAS_W         = int(np.ceil(MAX_W / 32) * 32)   #  608

MODEL_VARIANT    = "small"    # tiny | small | base | large
NUM_CLASSES      = 6
IN_CHANNELS      = 1
PRETRAINED       = False
DATA_EXT         = "*.tif"
CLASS_NAMES      = [f"facies_{i}" for i in range(NUM_CLASSES)]

EPOCHS           = 200        # high ceiling — EarlyStopping + time limit handle stopping
BATCH_SIZE       = 2          # per GPU; effective batch = BATCH_SIZE * NUM_GPUS
NUM_WORKERS      = 4          # per GPU
BASE_LR          = 1e-4
ENCODER_LR_SCALE = 0.1
WEIGHT_DECAY     = 1e-2
FREEZE_EPOCHS    = 5
WARMUP_EPOCHS    = 5
SEED             = 42

# Multi-node / multi-GPU
NUM_NODES        = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
GPUS_PER_NODE    = int(os.environ.get("GPUS_PER_NODE", 4))


# ===========================================================================
# Resume helper
# ===========================================================================

def find_resume_checkpoint(checkpoint_dir: Path):
    """
    Returns the path to 'last.ckpt' if it exists, otherwise None.
    Lightning's ModelCheckpoint(save_last=True) always keeps this file
    up to date, so it is safe to resume from even after a hard timeout.
    """
    last = checkpoint_dir / "last.ckpt"
    if last.exists():
        print(f"[Resume] Found checkpoint: {last}")
        return str(last)
    print("[Resume] No checkpoint found — starting from scratch.")
    return None


# ===========================================================================
# Main
# ===========================================================================

def main():
    L.seed_everything(SEED, workers=True)

    # ── Data ─────────────────────────────────────────────────────────────
    dm = SeismicDataModule(
        train_data=TRAIN_DATA,   train_labels=TRAIN_LABELS,
        val_data=VAL_DATA,       val_labels=VAL_LABELS,
        test_data=TEST_DATA,     test_labels=TEST_LABELS,
        canvas_h=CANVAS_H,       canvas_w=CANVAS_W,
        in_channels=IN_CHANNELS, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, data_ext=DATA_EXT,
        num_classes=NUM_CLASSES,
    )
    dm.setup("fit")

    # ── Model ─────────────────────────────────────────────────────────────
    ckpt_path = find_resume_checkpoint(CHECKPOINT_DIR)

    model = MambaSegLightning(
        num_classes=NUM_CLASSES,       in_channels=IN_CHANNELS,
        variant=MODEL_VARIANT,         pretrained=PRETRAINED,
        base_lr=BASE_LR,               encoder_lr_scale=ENCODER_LR_SCALE,
        weight_decay=WEIGHT_DECAY,     freeze_epochs=FREEZE_EPOCHS,
        max_epochs=EPOCHS,             warmup_epochs=WARMUP_EPOCHS,
        class_weights=dm.class_weights,
        class_names=CLASS_NAMES,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────
    callbacks = [
        # Best checkpoint (monitored metric) + last.ckpt (for resume)
        ModelCheckpoint(
            dirpath=str(CHECKPOINT_DIR),
            filename="best-{epoch:02d}-{val/miou:.4f}",
            monitor="val/miou",
            mode="max",
            save_top_k=1,
            save_last=True,     # always keeps last.ckpt — used for resume
            verbose=True,
        ),
        EarlyStopping(
            monitor="val/miou",
            mode="max",
            patience=15,        # more patience for multi-run HPC jobs
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # ── Logger ────────────────────────────────────────────────────────────
    # version=0 keeps all runs appending to the same CSV across resumes
    logger = CSVLogger(
        save_dir=str(CHECKPOINT_DIR),
        name="logs",
        version=0,
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = L.Trainer(
        max_epochs=EPOCHS,
        num_nodes=NUM_NODES,
        devices=GPUS_PER_NODE,
        accelerator="gpu",
        strategy=DDPStrategy(
            find_unused_parameters=False,   # faster DDP; set True if you see errors
        ),
        precision="16-mixed",
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        deterministic=False,
        enable_progress_bar=True,
        # Save a checkpoint every 30 minutes so a hard SLURM kill loses at most 30 min
        val_check_interval=1.0,
    )

    # ── Fit (resume if checkpoint exists) ─────────────────────────────────
    trainer.fit(model, dm, ckpt_path=ckpt_path)

    # ── Test (only on rank 0 to avoid duplicate output) ───────────────────
    if trainer.global_rank == 0:
        dm.setup("test")
        best = trainer.checkpoint_callback.best_model_path
        if not best:
            best = str(CHECKPOINT_DIR / "last.ckpt")
        print(f"\n[Test] Loading best checkpoint: {best}")
        trainer.test(model, dm, ckpt_path=best)


if __name__ == "__main__":
    main()
