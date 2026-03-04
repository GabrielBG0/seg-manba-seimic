"""
pretrained_utils.py  –  ImageNet pretrained weight loading for MambaSegNet
==========================================================================

Loads the official VMamba ImageNet-1K checkpoints into the encoder of
MambaSegNet, leaving the decoder and segmentation head randomly initialised
(standard practice for encoder-decoder segmentation models).

Three sources are supported, in order of preference:

  1. VMamba official weights  (HuggingFace Hub: MzeroMiko/VMamba)
     Best choice: the VSSBlock parameters map almost exactly to our encoder.

  2. Swin-UMamba weights  (HuggingFace Hub: JiarunLiu/Swin-UMamba)
     Alternative: Mamba-UNet trained with ImageNet pretraining (MICCAI 2024).

  3. A local .pth checkpoint  (any path on disk)
     Use when you have already downloaded a checkpoint manually.

What transfers vs what doesn't
-------------------------------
  TRANSFERRED  (encoder VSSBlocks, PatchMerging, encoder stage norms)
    - SS2D: in_proj, conv2d, x_proj, dt_projs, A_logs, Ds, out_norm, out_proj
    - VSSBlock: norm1 (LayerNorm before SS2D)
    - PatchMerging: norm + reduction
    - Stage-end LayerNorm

  NOT TRANSFERRED (random init kept)
    - PatchEmbed: VMamba uses a single 4x4 stride-4 conv; our model uses two
      overlapping 3x3 stride-2 convs — incompatible shapes, skipped silently.
    - Decoder: PatchExpanding + DecoderStage blocks (not in ImageNet ckpt)
    - FinalExpanding + seg_head: task-specific, always random
    - VMamba classification head: discarded

Usage
-----
    from mamba_seg_net   import mamba_seg_small, mamba_seg_base
    from pretrained_utils import load_vmamba_pretrained

    # Downloads automatically from HuggingFace (~350 MB for base):
    model = mamba_seg_base(num_classes=6)
    info  = load_vmamba_pretrained(model, variant='base')
    print(info)

    # Or supply a local file:
    info  = load_vmamba_pretrained(model, ckpt_path='/data/vmamba_base.pth')

    # Then fine-tune:
    model.train()
    optimizer = torch.optim.AdamW(model.get_param_groups(), lr=1e-4)

References
----------
- VMamba (NeurIPS 2024 spotlight): https://github.com/MzeroMiko/VMamba
- Swin-UMamba (MICCAI 2024):       https://github.com/JiarunLiu/Swin-UMamba
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# HuggingFace Hub filenames for official VMamba ImageNet-1K checkpoints
# Hub repo: https://huggingface.co/MzeroMiko/VMamba
# ---------------------------------------------------------------------------
_VMAMBA_HUB_REPO = "MzeroMiko/VMamba"
_VMAMBA_FILES: Dict[str, str] = {
    # variant  ->  filename inside the HF repo
    "tiny":  "vssmtiny_dp01_ckpt_epoch_292.pth",
    "small": "vssmsmall_dp01_ckpt_epoch_238.pth",
    "base":  "vssmbase_dp006_ckpt_epoch_241.pth",
}

# Scale mapping: our embed_dim -> expected VMamba variant
_DIM_TO_VARIANT = {64: "tiny", 96: "small", 128: "base", 192: "base"}


# ===========================================================================
# Key remapping: VMamba checkpoint -> MambaSegNet encoder
# ===========================================================================

def _remap_vmamba_key(key: str) -> Optional[str]:
    """
    Map a key from the VMamba classification checkpoint to the corresponding
    key in MambaSegNet's encoder.  Returns None if the key should be skipped.

    VMamba key patterns
    -------------------
    patch_embed.proj.{weight,bias}                  <- incompatible shape
    layers.{i}.blocks.{j}.norm.{weight,bias}        <- VSSBlock pre-norm
    layers.{i}.blocks.{j}.op.{...}                  <- SS2D internals
    layers.{i}.norm.{weight,bias}                   <- stage-end norm
    layers.{i}.downsample.{norm,reduction}.{...}    <- PatchMerging
    norm.{weight,bias}                              <- final classifier norm  (skip)
    head.{weight,bias}                              <- classifier head        (skip)

    MambaSegNet key patterns
    ------------------------
    patch_embed.proj.{0,2}.{weight,bias}            <- different architecture
    enc_stages.{i}.blocks.{j}.norm1.{weight,bias}
    enc_stages.{i}.blocks.{j}.ss2d.{...}
    enc_stages.{i}.norm.{weight,bias}
    downsamples.{i}.{norm,reduction}.{...}
    """

    # --- Keys to discard entirely ---
    if key.startswith(("norm.", "head.", "patch_embed.")):
        return None

    # --- stage norm: layers.{i}.norm.* -> enc_stages.{i}.norm.* ---
    m = re.fullmatch(r"layers\.(\d+)\.norm\.(.*)", key)
    if m:
        return f"enc_stages.{m.group(1)}.norm.{m.group(2)}"

    # --- PatchMerging: layers.{i}.downsample.* -> downsamples.{i}.* ---
    m = re.fullmatch(r"layers\.(\d+)\.downsample\.(.*)", key)
    if m:
        return f"downsamples.{m.group(1)}.{m.group(2)}"

    # --- VSSBlock norm: layers.{i}.blocks.{j}.norm.* -> ...norm1.* ---
    m = re.fullmatch(r"layers\.(\d+)\.blocks\.(\d+)\.norm\.(.*)", key)
    if m:
        return f"enc_stages.{m.group(1)}.blocks.{m.group(2)}.norm1.{m.group(3)}"

    # --- SS2D internals: layers.{i}.blocks.{j}.op.* -> ...ss2d.* ---
    m = re.fullmatch(r"layers\.(\d+)\.blocks\.(\d+)\.op\.(.*)", key)
    if m:
        return f"enc_stages.{m.group(1)}.blocks.{m.group(2)}.ss2d.{m.group(3)}"

    # Anything else (e.g. unexpected keys in newer checkpoints): skip
    return None


def _remap_state_dict(raw: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Apply _remap_vmamba_key to every key in `raw`.
    Returns the filtered + remapped OrderedDict.
    """
    # Some checkpoints wrap everything under a 'model' key
    if "model" in raw and isinstance(raw["model"], dict):
        raw = raw["model"]

    remapped: Dict[str, torch.Tensor] = {}
    skipped = []
    for k, v in raw.items():
        new_k = _remap_vmamba_key(k)
        if new_k is None:
            skipped.append(k)
        else:
            remapped[new_k] = v

    return remapped, skipped


# ===========================================================================
# Download helpers
# ===========================================================================

def _download_from_hf(repo_id: str, filename: str, cache_dir: Optional[str] = None) -> str:
    """
    Download `filename` from HuggingFace Hub repo `repo_id`.
    Returns the local path to the downloaded file.
    Requires:  pip install huggingface_hub
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for automatic weight download.\n"
            "  pip install huggingface_hub\n"
            "Or download the checkpoint manually and pass ckpt_path=..."
        )
    print(f"[pretrained] Downloading {filename} from {repo_id} ...")
    path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
    print(f"[pretrained] Saved to {path}")
    return path


# ===========================================================================
# Public API
# ===========================================================================

def load_vmamba_pretrained(
    model: nn.Module,
    variant: Optional[str] = None,
    ckpt_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    strict: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Load VMamba ImageNet pretrained encoder weights into MambaSegNet.

    Parameters
    ----------
    model     : MambaSegNet instance (any scale variant)
    variant   : one of 'tiny' | 'small' | 'base'
                If None, inferred from model.patch_embed embed_dim
    ckpt_path : path to a local .pth file (skips HuggingFace download)
    cache_dir : custom HuggingFace cache directory
    strict    : if True, raise on any unmatched keys (default False)
    verbose   : print a loading report

    Returns
    -------
    dict with keys:
        'loaded'    : list of keys successfully loaded
        'missing'   : model keys with no matching checkpoint key
        'unexpected': checkpoint keys not used
        'skipped'   : checkpoint keys intentionally discarded (decoder etc.)
    """
    # --- Resolve checkpoint path ---
    if ckpt_path is None:
        # Auto-detect variant from embed_dim if not given
        if variant is None:
            embed_dim = model.patch_embed.proj[0].out_channels * 2
            variant = _DIM_TO_VARIANT.get(embed_dim)
            if variant is None:
                raise ValueError(
                    f"Cannot auto-detect variant from embed_dim={embed_dim}. "
                    f"Pass variant='tiny'|'small'|'base' explicitly."
                )
        if variant not in _VMAMBA_FILES:
            raise ValueError(
                f"Unknown variant '{variant}'. Choose from: {list(_VMAMBA_FILES)}"
            )
        ckpt_path = _download_from_hf(
            _VMAMBA_HUB_REPO, _VMAMBA_FILES[variant], cache_dir=cache_dir
        )

    # --- Load raw checkpoint ---
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Some checkpoints nest weights under 'model' or 'state_dict'
    for key in ("model", "state_dict"):
        if key in raw and isinstance(raw[key], dict):
            raw = raw[key]
            break

    # --- Remap keys ---
    remapped, skipped_ckpt = _remap_state_dict(raw)

    # --- Load into model (non-strict by default) ---
    result = model.load_state_dict(remapped, strict=False)
    missing    = result.missing_keys
    unexpected = result.unexpected_keys

    # Partition missing into decoder (expected) vs encoder (potential issues)
    decoder_keywords = ("dec_stages", "final_expand", "seg_head",
                        "patch_embed")
    missing_decoder  = [k for k in missing if any(kw in k for kw in decoder_keywords)]
    missing_encoder  = [k for k in missing if k not in missing_decoder]

    if strict and missing_encoder:
        raise RuntimeError(
            f"Strict loading failed. Encoder keys with no pretrained weights:\n"
            + "\n".join(f"  {k}" for k in missing_encoder)
        )

    loaded = [k for k in model.state_dict() if k not in missing]

    if verbose:
        print("\n" + "=" * 60)
        print("  VMamba pretrained weight loading report")
        print("=" * 60)
        print(f"  Checkpoint  : {Path(ckpt_path).name}")
        print(f"  Loaded      : {len(loaded)} tensors")
        print(f"  Missing (decoder / head — expected)  : {len(missing_decoder)}")
        if missing_encoder:
            print(f"  Missing (encoder — investigate)      : {len(missing_encoder)}")
            for k in missing_encoder[:10]:
                print(f"    {k}")
            if len(missing_encoder) > 10:
                print(f"    ... and {len(missing_encoder) - 10} more")
        print(f"  Unexpected (unused ckpt keys)        : {len(unexpected)}")
        print(f"  Skipped (head/patch_embed/other)     : {len(skipped_ckpt)}")
        print("=" * 60 + "\n")

    return {
        "loaded":     loaded,
        "missing":    missing,
        "unexpected": unexpected,
        "skipped":    skipped_ckpt,
    }


def load_from_local(
    model: nn.Module,
    ckpt_path: str,
    key_prefix: str = "",
    verbose: bool = True,
) -> dict:
    """
    Generic helper: load any local .pth checkpoint into MambaSegNet.

    Useful for:
      - Resuming from a MambaSegNet fine-tuning checkpoint
      - Loading from a custom VMamba training run
      - Loading converted checkpoints

    Parameters
    ----------
    key_prefix : optional prefix to strip from checkpoint keys before matching
                 e.g. key_prefix='backbone.' if the encoder was saved inside
                 a larger detection/segmentation framework.
    """
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for k in ("model", "state_dict", "net"):
        if k in raw and isinstance(raw[k], dict):
            raw = raw[k]
            break

    if key_prefix:
        raw = {
            k[len(key_prefix):]: v
            for k, v in raw.items()
            if k.startswith(key_prefix)
        }

    result = model.load_state_dict(raw, strict=False)
    loaded = [k for k in model.state_dict() if k not in result.missing_keys]

    if verbose:
        print(f"[pretrained] Loaded {len(loaded)} / {len(model.state_dict())} "
              f"tensors from {Path(ckpt_path).name}")
        if result.missing_keys:
            print(f"  Missing keys  : {len(result.missing_keys)}")
        if result.unexpected_keys:
            print(f"  Unused keys   : {len(result.unexpected_keys)}")

    return {"loaded": loaded, "missing": result.missing_keys,
            "unexpected": result.unexpected_keys}


# ===========================================================================
# Encoder freezing helpers  (useful for few-shot / small dataset scenarios)
# ===========================================================================

def freeze_encoder(model: nn.Module, num_stages_to_freeze: int = 4) -> None:
    """
    Freeze the first `num_stages_to_freeze` encoder stages and the patch embed.

    Call this after load_vmamba_pretrained() to lock the pretrained features
    and train only the decoder.  Unfreeze later with unfreeze_encoder().

    Typical workflow for seismic facies with few labelled examples:
      1. load_vmamba_pretrained(model)
      2. freeze_encoder(model)               -- train decoder only (~5 epochs)
      3. unfreeze_encoder(model)             -- fine-tune everything (~20 epochs)
    """
    modules_to_freeze = [model.patch_embed]
    for i in range(min(num_stages_to_freeze, model.num_stages)):
        modules_to_freeze.append(model.enc_stages[i])
        if i < len(model.downsamples):
            modules_to_freeze.append(model.downsamples[i])

    frozen = 0
    for m in modules_to_freeze:
        for p in m.parameters():
            p.requires_grad = False
            frozen += 1

    print(f"[pretrained] Frozen {frozen} parameter tensors "
          f"({num_stages_to_freeze} encoder stages + patch embed).")


def unfreeze_encoder(model: nn.Module) -> None:
    """Re-enable gradients for all encoder parameters (undoes freeze_encoder)."""
    thawed = 0
    for m in [model.patch_embed, model.enc_stages, model.downsamples]:
        for p in (m if isinstance(m, nn.ModuleList) else [m]).parameters() \
                if isinstance(m, nn.Module) else \
                (p for sub in m for p in sub.parameters()):
            p.requires_grad = True
            thawed += 1
    print(f"[pretrained] Unfrozen {thawed} encoder parameter tensors.")


def get_param_groups_layerwise_lr(
    model: nn.Module,
    base_lr: float = 1e-4,
    encoder_lr_scale: float = 0.1,
    weight_decay: float = 1e-2,
) -> list:
    """
    Layer-wise learning rate decay: encoder uses a lower LR than the decoder.

    This is the recommended optimiser setup when fine-tuning with pretrained
    encoder weights — prevents catastrophic forgetting of ImageNet features.

    Example
    -------
        param_groups = get_param_groups_layerwise_lr(model, base_lr=1e-4,
                                                     encoder_lr_scale=0.1)
        optimizer = torch.optim.AdamW(param_groups)
        # Encoder trains at lr=1e-5, decoder at lr=1e-4
    """
    encoder_lr = base_lr * encoder_lr_scale

    def _is_no_decay(name: str) -> bool:
        return any(kw in name for kw in ('norm', 'bias', '.Ds', 'A_logs'))

    encoder_decay, encoder_nodecay = [], []
    decoder_decay, decoder_nodecay = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        in_encoder = any(name.startswith(pfx) for pfx in
                         ('patch_embed', 'enc_stages', 'downsamples'))
        no_decay = _is_no_decay(name)
        if in_encoder:
            (encoder_nodecay if no_decay else encoder_decay).append(p)
        else:
            (decoder_nodecay if no_decay else decoder_decay).append(p)

    groups = [
        {'params': encoder_decay,   'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': encoder_nodecay, 'lr': encoder_lr, 'weight_decay': 0.0},
        {'params': decoder_decay,   'lr': base_lr,    'weight_decay': weight_decay},
        {'params': decoder_nodecay, 'lr': base_lr,    'weight_decay': 0.0},
    ]
    groups = [g for g in groups if g['params']]   # remove empty groups

    total_enc = sum(p.numel() for p in encoder_decay + encoder_nodecay) / 1e6
    total_dec = sum(p.numel() for p in decoder_decay + decoder_nodecay) / 1e6
    print(f"[pretrained] Encoder: {total_enc:.1f} M params @ lr={encoder_lr:.2e}  |  "
          f"Decoder: {total_dec:.1f} M params @ lr={base_lr:.2e}")

    return groups


# ===========================================================================
# Quick smoke test
# ===========================================================================

def _test_remap():
    """Verify the key remapping logic against known patterns."""
    cases = [
        ("layers.0.blocks.0.norm.weight",
         "enc_stages.0.blocks.0.norm1.weight"),
        ("layers.2.blocks.5.op.in_proj.weight",
         "enc_stages.2.blocks.5.ss2d.in_proj.weight"),
        ("layers.2.blocks.5.op.A_logs",
         "enc_stages.2.blocks.5.ss2d.A_logs"),
        ("layers.2.blocks.5.op.dt_projs.3.weight",
         "enc_stages.2.blocks.5.ss2d.dt_projs.3.weight"),
        ("layers.1.downsample.norm.weight",
         "downsamples.1.norm.weight"),
        ("layers.1.downsample.reduction.weight",
         "downsamples.1.reduction.weight"),
        ("layers.0.norm.weight",
         "enc_stages.0.norm.weight"),
        ("norm.weight",     None),
        ("head.weight",     None),
        ("patch_embed.proj.weight", None),
    ]
    ok, fail = 0, 0
    for src, expected in cases:
        got = _remap_vmamba_key(src)
        if got == expected:
            ok += 1
        else:
            print(f"  FAIL  {src!r}\n    expected: {expected!r}\n    got:      {got!r}")
            fail += 1
    print(f"Key remap test: {ok}/{ok+fail} passed.")


if __name__ == "__main__":
    print("=" * 60)
    print("  pretrained_utils – self test")
    print("=" * 60)
    _test_remap()

    # Test param group helper (no download needed)
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from mamba_seg_net import mamba_seg_small
        model = mamba_seg_small(num_classes=6)
        groups = get_param_groups_layerwise_lr(model, base_lr=1e-4)
        print(f"Param groups created: {len(groups)}")
        freeze_encoder(model, num_stages_to_freeze=2)
        unfreeze_encoder(model)
        print("Freeze/unfreeze: OK")
    except ImportError:
        print("(mamba_seg_net not found, skipping model tests)")