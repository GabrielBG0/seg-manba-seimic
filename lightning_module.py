"""
lightning_module.py  —  PyTorch Lightning wrappers for MambaSegNet
==================================================================

Provides two classes:

  SeismicDataModule   — LightningDataModule
      Wraps SeismicFaciesDataset for train / val / test splits.
      Handles class-weight computation and DataLoader creation.

  MambaSegLightning   — LightningModule
      Wraps MambaSegNet with:
        - Dice+CE loss with class weights
        - Two-phase LR: encoder frozen for freeze_epochs, then layer-wise LR decay
        - Cosine annealing with linear warm-up
        - Per-epoch mIoU and per-class IoU logging (compatible with TensorBoard,
          WandB, CSV — whichever logger is attached to the Trainer)
        - Best checkpoint saving via ModelCheckpoint callback
        - Optional VMamba ImageNet pretraining

Usage (notebook or script)
--------------------------
    from lightning_module import SeismicDataModule, MambaSegLightning
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

    dm = SeismicDataModule(
        train_data=TRAIN_DATA, train_labels=TRAIN_LABELS,
        val_data=VAL_DATA,     val_labels=VAL_LABELS,
        test_data=TEST_DATA,   test_labels=TEST_LABELS,
        canvas_h=CANVAS_H,     canvas_w=CANVAS_W,
        in_channels=IN_CHANNELS,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        data_ext="*.tif",
    )
    dm.setup("fit")

    model = MambaSegLightning(
        num_classes=NUM_CLASSES,
        in_channels=IN_CHANNELS,
        variant=MODEL_VARIANT,
        pretrained=PRETRAINED,
        base_lr=BASE_LR,
        encoder_lr_scale=ENCODER_LR_SCALE,
        weight_decay=WEIGHT_DECAY,
        freeze_epochs=FREEZE_EPOCHS,
        max_epochs=EPOCHS,
        class_weights=dm.class_weights,
        class_names=CLASS_NAMES,
    )

    trainer = L.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu", devices=1,
        precision="16-mixed",
        callbacks=[
            ModelCheckpoint(monitor="val/miou", mode="max", save_top_k=1,
                            filename="best-{epoch:02d}-{val/miou:.4f}"),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        log_every_n_steps=10,
    )

    trainer.fit(model, dm)
    trainer.test(model, dm)
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import lightning as L
import tifffile

from mamba_seg_net import (
    DiceCELoss,
    mamba_seg_base,
    mamba_seg_large,
    mamba_seg_small,
    mamba_seg_tiny,
)
from pretrained_utils import (
    freeze_encoder,
    get_param_groups_layerwise_lr,
    load_vmamba_pretrained,
    unfreeze_encoder,
)

# ===========================================================================
# 1.  Dataset  (unchanged from notebook — reproduced here for self-containment)
# ===========================================================================

class SeismicFaciesDataset(Dataset):
    """
    Paired (amplitude .tif, facies label .png) seismic slice dataset.

    All slices are padded to (canvas_h, canvas_w) so mixed inline/crossline
    batches work with the DataLoader.  Padded label pixels are set to
    IGNORE_INDEX so the loss never penalises them.
    """

    IGNORE_INDEX = -1

    _AUGMENT_FLIP_H = 0.5
    _AUGMENT_FLIP_V = 0.3
    _AUGMENT_NOISE  = 0.3
    _AUGMENT_SCALE  = 0.3

    def __init__(
        self,
        data_dir: Path,
        label_dir: Path,
        canvas_h: int,
        canvas_w: int,
        in_channels: int = 1,
        augment: bool = False,
        num_classes: int = 6,
        data_ext: str = "*.tif",
    ):
        assert canvas_h % 32 == 0, f"canvas_h={canvas_h} must be divisible by 32"
        assert canvas_w % 32 == 0, f"canvas_w={canvas_w} must be divisible by 32"

        self.data_dir    = Path(data_dir)
        self.label_dir   = Path(label_dir)
        self.canvas_h    = canvas_h
        self.canvas_w    = canvas_w
        self.in_channels = in_channels
        self.augment     = augment
        self.num_classes = num_classes

        self.samples: List = []
        il_count = xl_count = 0
        for tif in sorted(self.data_dir.glob(data_ext)):
            lbl = self.label_dir / (tif.stem + ".png")
            if lbl.exists():
                self.samples.append((tif, lbl))
                if tif.stem.startswith("il"):
                    il_count += 1
                elif tif.stem.startswith("xl"):
                    xl_count += 1
            else:
                print(f"[Dataset] WARNING: no label for {tif.name}, skipping.")

        if not self.samples:
            raise FileNotFoundError(
                f"No matched pairs in {self.data_dir} / {self.label_dir}"
            )
        suffix = f" (il={il_count}, xl={xl_count})" if il_count or xl_count else ""
        print(f"[Dataset] {len(self.samples)} samples{suffix} | "
              f"canvas {canvas_h}×{canvas_w}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tif_path, lbl_path = self.samples[idx]

        amp = tifffile.imread(str(tif_path)).astype(np.float32)
        if amp.ndim == 3:
            amp = amp[0] if amp.shape[0] < amp.shape[-1] else amp[..., 0]

        lbl = np.array(Image.open(lbl_path), dtype=np.int64)
        if lbl.ndim == 3:
            lbl = lbl[..., 0]

        amp = (amp - amp.mean()) / (amp.std() + 1e-8)

        if self.augment:
            amp, lbl = self._augment(amp, lbl)

        amp, lbl = self._pad_to_canvas(amp, lbl)

        amp_t = torch.from_numpy(amp).unsqueeze(0)
        if self.in_channels == 3:
            amp_t = amp_t.repeat(3, 1, 1)
        lbl_t = torch.from_numpy(lbl).long()

        return amp_t, lbl_t, tif_path.name

    def _pad_to_canvas(self, amp, lbl):
        H, W = amp.shape
        CH, CW = self.canvas_h, self.canvas_w
        if H > CH:
            y0 = (H - CH) // 2; amp = amp[y0:y0+CH]; lbl = lbl[y0:y0+CH]; H = CH
        if W > CW:
            x0 = (W - CW) // 2; amp = amp[:, x0:x0+CW]; lbl = lbl[:, x0:x0+CW]; W = CW
        ph, pw = CH - H, CW - W
        if ph > 0 or pw > 0:
            amp = np.pad(amp, ((0, ph), (0, pw)), mode="reflect")
            lbl = np.pad(lbl, ((0, ph), (0, pw)),
                         mode="constant", constant_values=self.IGNORE_INDEX)
        return amp, lbl

    def _augment(self, amp, lbl):
        if random.random() < self._AUGMENT_FLIP_H:
            amp, lbl = np.fliplr(amp), np.fliplr(lbl)
        if random.random() < self._AUGMENT_FLIP_V:
            amp, lbl = np.flipud(amp), np.flipud(lbl)
        if random.random() < self._AUGMENT_NOISE:
            amp = amp + np.random.normal(0, 0.05, amp.shape).astype(np.float32)
        if random.random() < self._AUGMENT_SCALE:
            amp = amp * random.uniform(0.7, 1.3)
        return amp.copy(), lbl.copy()


# ===========================================================================
# 2.  LightningDataModule
# ===========================================================================

class SeismicDataModule(L.LightningDataModule):
    """
    DataModule wrapping train / val / test SeismicFaciesDatasets.

    Also computes inverse-median-frequency class weights over the training set
    (exposed as self.class_weights for use in the loss function).

    Parameters
    ----------
    train_data / train_labels : paths to training data and annotation folders
    val_data   / val_labels   : validation folders
    test_data  / test_labels  : test folders
    canvas_h / canvas_w       : padded spatial size (must be div by 32)
    in_channels               : 1 for raw amplitude, 3 for pretrained encoder
    batch_size                : training batch size (val/test always use 1)
    num_workers               : DataLoader workers
    data_ext                  : glob pattern for data files (default "*.tif")
    num_classes               : number of facies classes
    """

    def __init__(
        self,
        train_data: Path, train_labels: Path,
        val_data:   Path, val_labels:   Path,
        test_data:  Path, test_labels:  Path,
        canvas_h: int,
        canvas_w: int,
        in_channels: int = 1,
        batch_size: int = 4,
        num_workers: int = 4,
        data_ext: str = "*.tif",
        num_classes: int = 6,
    ):
        super().__init__()
        self.train_data    = Path(train_data)
        self.train_labels  = Path(train_labels)
        self.val_data      = Path(val_data)
        self.val_labels    = Path(val_labels)
        self.test_data     = Path(test_data)
        self.test_labels   = Path(test_labels)
        self.canvas_h      = canvas_h
        self.canvas_w      = canvas_w
        self.in_channels   = in_channels
        self.batch_size    = batch_size
        self.num_workers   = num_workers
        self.data_ext      = data_ext
        self.num_classes   = num_classes
        self.class_weights: Optional[torch.Tensor] = None

    def setup(self, stage: Optional[str] = None):
        if stage in ("fit", None):
            self.train_ds = SeismicFaciesDataset(
                self.train_data, self.train_labels,
                self.canvas_h, self.canvas_w,
                self.in_channels, augment=True,
                num_classes=self.num_classes, data_ext=self.data_ext,
            )
            self.val_ds = SeismicFaciesDataset(
                self.val_data, self.val_labels,
                self.canvas_h, self.canvas_w,
                self.in_channels, augment=False,
                num_classes=self.num_classes, data_ext=self.data_ext,
            )
            if self.class_weights is None:
                self.class_weights = self._compute_class_weights(self.train_ds)

        if stage in ("test", None):
            self.test_ds = SeismicFaciesDataset(
                self.test_data, self.test_labels,
                self.canvas_h, self.canvas_w,
                self.in_channels, augment=False,
                num_classes=self.num_classes, data_ext=self.data_ext,
            )

    def _compute_class_weights(self, dataset) -> torch.Tensor:
        print("[DataModule] Computing class weights over training set...")
        counts = np.zeros(self.num_classes, dtype=np.int64)
        for _, lbl_t, _ in dataset:
            lbl = lbl_t.numpy().ravel()
            valid = lbl[lbl >= 0]
            for c in range(self.num_classes):
                counts[c] += (valid == c).sum()
        freq   = counts / (counts.sum() + 1e-8)
        median = np.median(freq[freq > 0])
        w = np.where(freq > 0, median / freq, 0.0).astype(np.float32)
        print(f"[DataModule] Class weights: {np.round(w, 3)}")
        return torch.tensor(w)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers,
            pin_memory=True, drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=1,
            shuffle=False, num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=1,
            shuffle=False, num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


# ===========================================================================
# 3.  LightningModule
# ===========================================================================

_FACTORIES = {
    "tiny":  mamba_seg_tiny,
    "small": mamba_seg_small,
    "base":  mamba_seg_base,
    "large": mamba_seg_large,
}


class MambaSegLightning(L.LightningModule):
    """
    LightningModule wrapping MambaSegNet for seismic facies segmentation.

    Training schedule
    -----------------
    Phase 1 (epochs 1 … freeze_epochs):
        Encoder frozen, only decoder trains at base_lr.
    Phase 2 (epochs freeze_epochs+1 … max_epochs):
        All parameters train; encoder at base_lr * encoder_lr_scale,
        decoder at base_lr (layer-wise LR decay).
    Both phases use cosine annealing with a linear warm-up over warmup_epochs.

    Metrics logged every epoch
    --------------------------
    train/loss, train/miou
    val/loss,   val/miou,   val/iou_<class_name>  (per-class)
    test/loss,  test/miou,  test/iou_<class_name>

    Parameters
    ----------
    num_classes       : number of facies classes
    in_channels       : 1 (amplitude) or 3 (RGB / pretrained)
    variant           : 'tiny' | 'small' | 'base' | 'large'
    pretrained        : load VMamba ImageNet encoder weights (needs in_channels=3)
    base_lr           : decoder (and default) learning rate
    encoder_lr_scale  : encoder lr = base_lr * encoder_lr_scale
    weight_decay      : AdamW weight decay
    freeze_epochs     : number of epochs to keep encoder frozen
    max_epochs        : total training epochs (needed for scheduler)
    warmup_epochs     : linear warm-up duration
    class_weights     : (num_classes,) tensor of per-class loss weights
    class_names       : list of class name strings for metric logging
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        variant: str = "small",
        pretrained: bool = False,
        base_lr: float = 1e-4,
        encoder_lr_scale: float = 0.1,
        weight_decay: float = 1e-2,
        freeze_epochs: int = 5,
        max_epochs: int = 50,
        warmup_epochs: int = 5,
        class_weights: Optional[torch.Tensor] = None,
        class_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        self.class_names   = class_names or [f"facies_{i}" for i in range(num_classes)]
        self.num_classes   = num_classes
        self.freeze_epochs = freeze_epochs

        # ── Model ────────────────────────────────────────────────────────────
        self.net = _FACTORIES[variant](
            num_classes=num_classes,
            in_channels=in_channels,
        )

        if pretrained and in_channels == 3:
            load_vmamba_pretrained(self.net, variant=variant)
        elif pretrained and in_channels != 3:
            print("[Lightning] PRETRAINED skipped: requires in_channels=3.")

        if pretrained and in_channels == 3 and freeze_epochs > 0:
            freeze_encoder(self.net)
            print(f"[Lightning] Encoder frozen for first {freeze_epochs} epoch(s).")

        # ── Loss ─────────────────────────────────────────────────────────────
        self.criterion = DiceCELoss(
            ce_weight=1.0, dice_weight=1.0,
            class_weights=class_weights,
            ignore_index=-1,
        )

        # ── Confusion matrix accumulators (reset each epoch) ─────────────────
        self._val_conf  = torch.zeros(num_classes, num_classes, dtype=torch.long)
        self._test_conf = torch.zeros(num_classes, num_classes, dtype=torch.long)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x):
        return self.net(x)

    # ------------------------------------------------------------------
    # Shared step
    # ------------------------------------------------------------------

    def _step(self, batch):
        amp, lbl, _ = batch
        logits = self(amp)
        loss   = self.criterion(logits, lbl)
        preds  = logits.argmax(dim=1)
        return loss, preds, lbl

    @staticmethod
    def _update_conf(conf, preds, targets):
        """Accumulate confusion matrix, ignoring IGNORE_INDEX=-1."""
        p = preds.cpu().view(-1)
        t = targets.cpu().view(-1)
        mask = t >= 0
        p, t = p[mask], t[mask]
        n = conf.shape[0]
        idx = t * n + p
        conf.view(-1).scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.long))

    @staticmethod
    def _miou_from_conf(conf):
        iou = []
        for c in range(conf.shape[0]):
            tp = conf[c, c].item()
            fp = conf[:, c].sum().item() - tp
            fn = conf[c, :].sum().item() - tp
            d  = tp + fp + fn
            iou.append(tp / d if d > 0 else float("nan"))
        valid = [v for v in iou if not np.isnan(v)]
        return float(np.mean(valid)) if valid else 0.0, iou

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def on_train_epoch_start(self):
        # Unfreeze encoder after freeze_epochs
        if (self.hparams.pretrained
                and self.hparams.in_channels == 3
                and self.current_epoch == self.freeze_epochs
                and self.freeze_epochs > 0):
            unfreeze_encoder(self.net)
            # Reconfigure optimizer with layer-wise LR now that encoder is active
            self.trainer.strategy.setup_optimizers(self.trainer)
            print(f"\n[Lightning] Epoch {self.current_epoch+1}: "
                  f"encoder unfrozen, layer-wise LR active.")

    def training_step(self, batch, batch_idx):
        loss, preds, lbl = self._step(batch)
        # Quick batch-level mIoU (not global — just for the progress bar)
        conf = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long)
        self._update_conf(conf, preds, lbl)
        miou, _ = self._miou_from_conf(conf)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/miou", miou, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def on_validation_epoch_start(self):
        self._val_conf.zero_()

    def validation_step(self, batch, batch_idx):
        loss, preds, lbl = self._step(batch)
        self._update_conf(self._val_conf, preds, lbl)
        self.log("val/loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        miou, iou_per_class = self._miou_from_conf(self._val_conf)
        self.log("val/miou", miou, prog_bar=True, sync_dist=True)
        for name, iou in zip(self.class_names, iou_per_class):
            if not np.isnan(iou):
                self.log(f"val/iou_{name}", iou, sync_dist=True)

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def on_test_epoch_start(self):
        self._test_conf.zero_()

    def test_step(self, batch, batch_idx):
        loss, preds, lbl = self._step(batch)
        self._update_conf(self._test_conf, preds, lbl)
        self.log("test/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def on_test_epoch_end(self):
        miou, iou_per_class = self._miou_from_conf(self._test_conf)
        self.log("test/miou", miou, sync_dist=True)
        print(f"\n{'='*50}")
        print(f"  Test mIoU : {miou:.4f}")
        print(f"{'='*50}")
        for name, iou in zip(self.class_names, iou_per_class):
            val = f"{iou:.4f}" if not np.isnan(iou) else "  n/a"
            self.log(f"test/iou_{name}", iou if not np.isnan(iou) else 0.0,
                     sync_dist=True)
            print(f"  {name:22s} {val}")
        print(f"{'='*50}\n")

    # ------------------------------------------------------------------
    # Optimiser + Scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        use_layerwise = (
            self.hparams.pretrained
            and self.hparams.in_channels == 3
            and self.current_epoch >= self.freeze_epochs
        )

        if use_layerwise:
            param_groups = get_param_groups_layerwise_lr(
                self.net,
                base_lr=self.hparams.base_lr,
                encoder_lr_scale=self.hparams.encoder_lr_scale,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            param_groups = self.net.get_param_groups(
                weight_decay=self.hparams.weight_decay
            )
            for g in param_groups:
                g.setdefault("lr", self.hparams.base_lr)

        optimizer = torch.optim.AdamW(param_groups)

        # Cosine annealing with linear warm-up
        warmup = self.hparams.warmup_epochs
        total  = self.hparams.max_epochs

        def lr_lambda(epoch):
            if epoch < warmup:
                return (epoch + 1) / warmup
            t = (epoch - warmup) / max(total - warmup, 1)
            return 0.5 * (1.0 + np.cos(np.pi * t))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }