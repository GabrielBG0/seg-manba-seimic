# train.py
import os, torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_float32_matmul_precision("high")

from pathlib import Path
import numpy as np
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from lightning_module import SeismicDataModule, MambaSegLightning

# ── Config ────────────────────────────────────────────────────────────────
DATASET_ROOT   = Path("dataset")
TRAIN_DATA     = DATASET_ROOT / "data"        / "train"
TRAIN_LABELS   = DATASET_ROOT / "annotations" / "train"
VAL_DATA       = DATASET_ROOT / "data"        / "val"
VAL_LABELS     = DATASET_ROOT / "annotations" / "val"
TEST_DATA      = DATASET_ROOT / "data"        / "test"
TEST_LABELS    = DATASET_ROOT / "annotations" / "test"
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

MAX_H, MAX_W     = 1006, 590
CANVAS_H         = int(np.ceil(MAX_H / 32) * 32)
CANVAS_W         = int(np.ceil(MAX_W / 32) * 32)
MODEL_VARIANT    = "small"
NUM_CLASSES      = 6
IN_CHANNELS      = 1
PRETRAINED       = False
DATA_EXT         = "*.tif"
EPOCHS           = 50
BATCH_SIZE       = 2
NUM_WORKERS      = 4
BASE_LR          = 1e-4
ENCODER_LR_SCALE = 0.1
WEIGHT_DECAY     = 1e-2
FREEZE_EPOCHS    = 5
WARMUP_EPOCHS    = 5
SEED             = 42
CLASS_NAMES      = [f"facies_{i}" for i in range(NUM_CLASSES)]

# ── Run ───────────────────────────────────────────────────────────────────
L.seed_everything(SEED, workers=True)

dm = SeismicDataModule(
    train_data=TRAIN_DATA, train_labels=TRAIN_LABELS,
    val_data=VAL_DATA,     val_labels=VAL_LABELS,
    test_data=TEST_DATA,   test_labels=TEST_LABELS,
    canvas_h=CANVAS_H,     canvas_w=CANVAS_W,
    in_channels=IN_CHANNELS, batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS, data_ext=DATA_EXT,
    num_classes=NUM_CLASSES,
)

model = MambaSegLightning(
    num_classes=NUM_CLASSES, in_channels=IN_CHANNELS,
    variant=MODEL_VARIANT,   pretrained=PRETRAINED,
    base_lr=BASE_LR,         encoder_lr_scale=ENCODER_LR_SCALE,
    weight_decay=WEIGHT_DECAY, freeze_epochs=FREEZE_EPOCHS,
    max_epochs=EPOCHS,       warmup_epochs=WARMUP_EPOCHS,
    class_weights=None,      class_names=CLASS_NAMES,
)

trainer = L.Trainer(
    max_epochs=EPOCHS,
    accelerator="gpu", devices=1,
    precision="16-mixed",
    gradient_clip_val=1.0,
    logger=CSVLogger(save_dir=str(CHECKPOINT_DIR), name="logs"),
    callbacks=[
        ModelCheckpoint(dirpath=CHECKPOINT_DIR,
                        filename="best-{epoch:02d}-{val/miou:.4f}",
                        monitor="val/miou", mode="max",
                        save_top_k=1, save_last=True),
        EarlyStopping(monitor="val/miou", mode="max", patience=10),
        LearningRateMonitor(logging_interval="epoch"),
    ],
    log_every_n_steps=10,
)

dm.setup("fit")
trainer.fit(model, dm)
dm.setup("test")
trainer.test(model, dm, ckpt_path=trainer.checkpoint_callback.best_model_path)