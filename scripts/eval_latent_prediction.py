"""Latent-prediction MSE evaluation at multiple horizons.

Phase 1 success criterion (pre-registered): graph encoder beats flat ViT encoder
on **latent prediction MSE at H=20** on held-out bouncing-balls trajectories.

This script computes that metric (and intermediate H=1, H=5) by:

  1. Loading a trained JEPA checkpoint.
  2. Loading the val split of a dataset (same split as training, derived from
     `train_split` in the training config — held-out means random_split with the
     training seed).
  3. For each held-out batch, encoding the full sequence to get
     ground-truth latents (`tgt_emb`).
  4. Autoregressively rolling the predictor forward `max(eval_horizons)` steps
     starting from the first `ctx_len` encoded frames as history.
  5. Computing per-step MSE between predicted and GT latents at each requested
     horizon H.

Notes
-----

* This is *not* CEM planning — it's open-loop latent rollout error, the
  cleanest pre-registered "world model accuracy" metric.

* For the graph encoder, absent slots are masked using `info["emb_mask"]`
  from the GT-encoded sequence (positions/mask are GT throughout training, so
  the mask is identical step-to-step within an episode and we can use the
  initial mask repeatedly).

* The training config uses `num_steps = num_preds + history_size = 4`, which
  is far too short for H=20 eval. We override
  `cfg.data.dataset.num_steps = ctx_len + max(eval_horizons)` at eval time
  before constructing the dataset.

CLI:

    python scripts/eval_latent_prediction.py \\
        --config-name lewm \\
        data=bouncing_balls \\
        ckpt_path=~/.stable-wm/lewm_weights.ckpt \\
        +eval_horizons=[1,5,20] \\
        +eval_n_episodes=200
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Optional

import hydra
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

# Make project root importable so `from jepa import JEPA`, etc. resolve when
# this script is invoked from anywhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from jepa import JEPA
from module import ARPredictor, Embedder, MLP
from utils import get_column_normalizer, get_img_preprocessor
from data.bouncing_balls_transform import BouncingBallsGraphTransform


# ---------------------------------------------------------------------------
# Model construction (mirrors train.py — kept in sync by intent)
# ---------------------------------------------------------------------------

def _build_world_model(cfg: DictConfig) -> JEPA:
    """Reconstruct the JEPA architecture from cfg, mirroring train.py.

    Returns an UNTRAINED model — caller is responsible for loading weights.
    """
    encoder_cfg = cfg.get("encoder", None)
    encoder_type = (encoder_cfg.type if encoder_cfg is not None else "vit")

    if encoder_type == "vit":
        encoder = spt.backbone.utils.vit_hf(
            (encoder_cfg.scale if encoder_cfg is not None else cfg.encoder_scale),
            patch_size=(encoder_cfg.patch_size if encoder_cfg is not None else cfg.patch_size),
            image_size=cfg.img_size,
            pretrained=False,
            use_mask_token=False,
        )
        hidden_dim = encoder.config.hidden_size
        embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    elif encoder_type == "graph":
        from models import GraphEncoder, SetPredictor  # noqa: F401 (SetPredictor used below)
        encoder = GraphEncoder(
            image_size=cfg.img_size,
            d_obj=encoder_cfg.d_obj,
            n_max=encoder_cfg.n_max,
            capacity=encoder_cfg.capacity,
            width_mode=encoder_cfg.width_mode,
        )
        if encoder_cfg.width_mode == "projected":
            embed_dim = 192
        elif encoder_cfg.width_mode == "native":
            embed_dim = int(encoder_cfg.d_obj)
        else:
            raise ValueError(
                f"Unknown encoder.width_mode={encoder_cfg.width_mode!r}"
            )
        hidden_dim = embed_dim
    else:
        raise ValueError(f"Unknown encoder.type={encoder_type!r}")

    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    if encoder_type == "graph":
        from models import SetPredictor
        predictor = SetPredictor(
            num_frames=cfg.wm.history_size,
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            n_max=encoder_cfg.n_max,
            **cfg.predictor,
        )
    else:
        predictor = ARPredictor(
            num_frames=cfg.wm.history_size,
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            **cfg.predictor,
        )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)

    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )
    pred_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    return JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
    )


# ---------------------------------------------------------------------------
# Checkpoint loading — supports two formats
# ---------------------------------------------------------------------------

def _load_checkpoint(ckpt_path: Path, cfg: DictConfig, device: torch.device) -> JEPA:
    """Load a trained JEPA from `ckpt_path`.

    Supports two on-disk formats:

      A. ModelObjectCallBack pickle (`*_object.ckpt`) — `torch.save(model, ...)`
         dumps a full `JEPA` instance. We `torch.load(weights_only=False)` it
         directly.

      B. spt.Manager Lightning checkpoint (`*_weights.ckpt`) — a dict with a
         `state_dict` whose keys are prefixed `model.` (because spt.Module
         wraps `model=JEPA`). We strip the prefix, drop the `sigreg.*` keys,
         and load into a freshly built JEPA.
    """
    obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    if isinstance(obj, JEPA):
        return obj.to(device)

    if isinstance(obj, torch.nn.Module):
        # Some other module class — try to use it directly. JEPA pickle from a
        # different module path will land here; warn and try `.to(device)`.
        return obj.to(device)

    if isinstance(obj, dict) and "state_dict" in obj:
        sd_full = obj["state_dict"]
        # Strip `model.` prefix, skip `sigreg.*` (eval doesn't need it).
        sd_jepa = {}
        for k, v in sd_full.items():
            if k.startswith("model."):
                sd_jepa[k[len("model.") :]] = v
            elif k.startswith("sigreg."):
                continue
            else:
                # Unknown prefix — keep as-is, let load_state_dict warn.
                sd_jepa[k] = v

        model = _build_world_model(cfg)
        missing, unexpected = model.load_state_dict(sd_jepa, strict=False)
        if missing:
            print(f"[load] missing keys ({len(missing)}): "
                  f"{missing[:5]}{' ...' if len(missing) > 5 else ''}",
                  file=sys.stderr)
        if unexpected:
            print(f"[load] unexpected keys ({len(unexpected)}): "
                  f"{unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}",
                  file=sys.stderr)
        return model.to(device)

    raise ValueError(
        f"Unrecognized checkpoint format at {ckpt_path}: type(obj)={type(obj)}. "
        "Expected a JEPA pickle or a dict with 'state_dict'."
    )


# ---------------------------------------------------------------------------
# Dataset construction (mirrors train.py — kept in sync by intent)
# ---------------------------------------------------------------------------

def _build_val_dataset(cfg: DictConfig):
    """Build the val split, using the same random_split + seed as train.py.

    Returns a torch DataLoader yielding batched samples with the *eval-time*
    sequence length (cfg.data.dataset.num_steps overridden upstream).
    """
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]

    data_transform_cfg = cfg.data.get("transform", None)
    skip_normalize = set((data_transform_cfg or {}).get("skip_normalize", []) or [])

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            if not hasattr(cfg.wm, f"{col}_dim"):
                setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))
            if col in skip_normalize:
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        if "action" not in cfg.data.dataset.keys_to_load:
            if data_transform_cfg is not None and "bouncing_balls_graph" in data_transform_cfg:
                cfg.wm.action_dim = int(data_transform_cfg.bouncing_balls_graph.action_dim)
            elif "action_dim" not in cfg.wm:
                cfg.wm.action_dim = 1

    if data_transform_cfg is not None and "bouncing_balls_graph" in data_transform_cfg:
        bb_kwargs = OmegaConf.to_container(
            data_transform_cfg.bouncing_balls_graph, resolve=True
        )
        transforms.append(BouncingBallsGraphTransform(**bb_kwargs))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    eval_bs = int(cfg.get("eval_batch_size", 16))
    loader_kwargs = dict(cfg.loader)
    loader_kwargs["batch_size"] = eval_bs
    # Eval-friendly loader settings: never shuffle, never drop_last.
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=eval_bs,
        num_workers=loader_kwargs.get("num_workers", 0),
        persistent_workers=False,
        pin_memory=loader_kwargs.get("pin_memory", False),
        shuffle=False,
        drop_last=False,
    )
    return val_loader


# ---------------------------------------------------------------------------
# Latent-prediction rollout
# ---------------------------------------------------------------------------

def _rollout_latents(
    model: JEPA,
    encoded: dict,
    ctx_len: int,
    n_steps: int,
    history_size: int,
) -> torch.Tensor:
    """Autoregressive latent rollout starting from the first ctx_len frames.

    Args:
        model: trained JEPA (eval mode).
        encoded: dict containing the full encoded sequence:
            - 'emb': (B, T, D) flat OR (B, T, N, D) graph
            - 'act_emb': (B, T, A_emb)
            - 'emb_mask': (B, T, N) bool, only present for graph
        ctx_len: number of context frames to seed from (typically 3).
        n_steps: how many future steps to predict.
        history_size: predictor's window size (typically 3).

    Returns:
        pred_emb of shape (B, n_steps, D) flat or (B, n_steps, N, D) graph,
        containing ONLY the n_steps predicted latents (not the seed context).
    """
    emb = encoded["emb"]            # (B, T, [N], D)
    act_emb = encoded["act_emb"]    # (B, T, A_emb)
    is_graph = (emb.ndim == 4)

    if is_graph:
        full_mask = encoded["emb_mask"]  # (B, T, N)
    else:
        full_mask = None

    # Seed history with the first ctx_len encoded latents.
    history = emb[:, :ctx_len].clone()              # (B, ctx_len, [N], D)
    if is_graph:
        # For bouncing balls (locked design): mask is identical across time
        # within an episode (same N visible balls). Use the most-recent slice
        # and extend on each rollout step.
        history_mask = full_mask[:, :ctx_len].clone()  # (B, ctx_len, N)

    preds = []
    for step in range(n_steps):
        # Slide the window: take the last `history_size` entries.
        emb_trunc = history[:, -history_size:]      # (B, HS, [N], D)
        # Action history: aligned with emb history (predictor consumes
        # action at step t to produce next-step prediction). The trained
        # forward pass uses `act_emb[:, :ctx_len]` aligned with
        # `emb[:, :ctx_len]`, so we mirror that: take act window aligned
        # with the latest history frames.
        # Indices: history covers original frames [step, step+1, ...,
        # step+ctx_len-1] in the global sequence. Action at the same
        # timesteps is required.
        # absolute index of the LAST frame in `history`:
        last_abs = ctx_len - 1 + step
        # window = last_abs - HS + 1 .. last_abs
        a_start = last_abs - history_size + 1
        a_end = last_abs + 1
        # Clip to available actions; if rollout runs longer than the loaded
        # sequence, repeat the last available action (autonomous-physics is
        # zero-action anyway, so this is a no-op there).
        T_avail = act_emb.size(1)
        a_end_c = min(a_end, T_avail)
        act_trunc = act_emb[:, max(a_start, 0): a_end_c]
        if act_trunc.size(1) < history_size:
            # Pad on the right with the last available action vector.
            pad_n = history_size - act_trunc.size(1)
            last_act = act_trunc[:, -1:].expand(-1, pad_n, *([-1] * (act_trunc.ndim - 2)))
            act_trunc = torch.cat([act_trunc, last_act], dim=1)

        if is_graph:
            mask_trunc = history_mask[:, -history_size:]
            pred_step = model.predict(emb_trunc, act_trunc, mask=mask_trunc)[:, -1:]
            # Extend mask with the most-recent slot pattern.
            history_mask = torch.cat([history_mask, history_mask[:, -1:]], dim=1)
        else:
            pred_step = model.predict(emb_trunc, act_trunc)[:, -1:]

        history = torch.cat([history, pred_step], dim=1)
        preds.append(pred_step)

    pred_seq = torch.cat(preds, dim=1)  # (B, n_steps, [N], D)
    return pred_seq


def _masked_mse_per_episode(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """MSE per episode (returns shape (B,)).

    For flat: pred,tgt of shape (B, D). MSE = mean over D.
    For graph: pred,tgt of shape (B, N, D), mask of shape (B, N).
        Per episode = sum of squared errors over real slots, divided by
        (n_real_slots * D). Slots where mask=False are excluded.
    """
    if pred.ndim == 2:
        return (pred - tgt).pow(2).mean(dim=-1)

    # Graph: (B, N, D)
    sq = (pred - tgt).pow(2)              # (B, N, D)
    if mask is None:
        return sq.mean(dim=(-1, -2))
    mask_f = mask.to(sq.dtype).unsqueeze(-1)  # (B, N, 1)
    sq_masked = sq * mask_f
    # Per-episode normalizer: real_slots * D
    D = sq.size(-1)
    n_real = mask_f.sum(dim=(-1, -2)).clamp_min(1.0)  # (B,) — counts ones in mask
    per_ep = sq_masked.sum(dim=(-1, -2)) / (n_real * D)
    return per_ep


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../config/train", config_name="lewm")
def run(cfg: DictConfig):
    # ---- Read eval-specific params (with defaults), and finalize cfg ----
    ckpt_path = cfg.get("ckpt_path", None)
    if ckpt_path is None:
        raise ValueError("ckpt_path is required (e.g. ckpt_path=~/.stable-wm/lewm_weights.ckpt)")
    ckpt_path = Path(os.path.expanduser(str(ckpt_path))).resolve()

    eval_horizons = list(cfg.get("eval_horizons", [1, 5, 20]))
    eval_n_episodes = int(cfg.get("eval_n_episodes", 200))
    # eval_split currently unused; reserved for future train/val/test triple.
    _ = cfg.get("eval_split", "val")
    eval_batch_size = int(cfg.get("eval_batch_size", 16))

    if not eval_horizons:
        raise ValueError("eval_horizons must be a non-empty list of positive ints")
    if min(eval_horizons) < 1:
        raise ValueError(f"eval_horizons must be >= 1, got {eval_horizons}")

    ctx_len = int(cfg.wm.history_size)
    history_size = int(cfg.wm.history_size)
    max_h = int(max(eval_horizons))
    required_T = ctx_len + max_h

    # Override num_steps so the dataset returns long-enough windows.
    with open_dict(cfg):
        cfg.data.dataset.num_steps = required_T
        cfg.eval_batch_size = eval_batch_size

    # ---- Device ----
    accel = cfg.trainer.get("accelerator", "gpu")
    if accel in ("cpu",) or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    print(f"[eval_latent_prediction] device={device}")
    print(f"[eval_latent_prediction] ctx_len={ctx_len} history_size={history_size}")
    print(f"[eval_latent_prediction] eval_horizons={eval_horizons} max_h={max_h}")
    print(f"[eval_latent_prediction] required sequence length T={required_T}")
    print(f"[eval_latent_prediction] eval_n_episodes={eval_n_episodes} batch={eval_batch_size}")
    print(f"[eval_latent_prediction] ckpt={ckpt_path}")

    # ---- Build val loader (must come before checkpoint load — sets wm dims) ----
    val_loader = _build_val_dataset(cfg)

    # ---- Load model ----
    model = _load_checkpoint(ckpt_path, cfg, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # ---- Run eval ----
    # Accumulate per-episode MSEs per horizon (so we can compute mean + stderr).
    per_h_episode_mses: dict[int, list[torch.Tensor]] = {h: [] for h in eval_horizons}
    seen = 0

    for batch in val_loader:
        if seen >= eval_n_episodes:
            break

        # Move batch tensors to device. NaN actions -> 0 (matches training).
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        if "action" in batch:
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)

        B_full = batch["pixels"].size(0)
        # Trim to remaining budget so per-H counts match.
        budget = eval_n_episodes - seen
        if B_full > budget:
            batch = {
                k: (v[:budget] if torch.is_tensor(v) else v) for k, v in batch.items()
            }
        B = batch["pixels"].size(0)

        # Encode the full sequence — gives both context-frame latents and
        # ground-truth latents at every step in [0, T).
        with torch.no_grad():
            encoded = model.encode(batch)
            pred_seq = _rollout_latents(
                model,
                encoded,
                ctx_len=ctx_len,
                n_steps=max_h,
                history_size=history_size,
            )  # (B, max_h, [N], D)

        # GT latents for the steps we predicted. Predicted step k (1..max_h)
        # corresponds to absolute frame index (ctx_len - 1 + k), i.e. starts at
        # ctx_len. So GT is encoded["emb"][:, ctx_len : ctx_len + max_h].
        gt_seq = encoded["emb"][:, ctx_len : ctx_len + max_h]
        if encoded.get("emb_mask") is not None:
            gt_mask_seq = encoded["emb_mask"][:, ctx_len : ctx_len + max_h]
        else:
            gt_mask_seq = None

        # Per-horizon: compare predicted at index (h-1) to GT at index (h-1).
        for h in eval_horizons:
            idx = h - 1
            pred_h = pred_seq[:, idx]                        # (B, [N], D)
            tgt_h = gt_seq[:, idx]                           # (B, [N], D)
            mask_h = gt_mask_seq[:, idx] if gt_mask_seq is not None else None
            per_ep_mse = _masked_mse_per_episode(pred_h, tgt_h, mask_h)  # (B,)
            per_h_episode_mses[h].append(per_ep_mse.detach().cpu())

        seen += B
        print(f"[eval_latent_prediction] processed {seen}/{eval_n_episodes} episodes")

    if seen == 0:
        raise RuntimeError("No episodes were evaluated — val loader was empty.")

    # ---- Aggregate ----
    rows = []
    for h in eval_horizons:
        per_ep = torch.cat(per_h_episode_mses[h])  # (n_ep,)
        n_ep = per_ep.numel()
        mean = per_ep.mean().item()
        std = per_ep.std(unbiased=True).item() if n_ep > 1 else 0.0
        sterr = std / math.sqrt(n_ep) if n_ep > 0 else 0.0
        rows.append({"horizon": h, "mse": mean, "stderr": sterr, "n_episodes": n_ep})

    # ---- Print markdown table ----
    print("\n## Latent prediction MSE\n")
    print("| Horizon | MSE | StdErr | N_episodes |")
    print("|--------:|----:|-------:|-----------:|")
    for r in rows:
        print(f"| H={r['horizon']} | {r['mse']:.4f} | {r['stderr']:.4f} | {r['n_episodes']} |")

    # ---- Persist JSON ----
    run_id = cfg.get("subdir") or ""
    # Resolve hydra interpolation manually if it's still a template string.
    if "${" in str(run_id):
        run_id = ""
    if not run_id:
        # Fall back to checkpoint stem so two eval runs on different ckpts
        # don't collide.
        run_id = ckpt_path.stem

    out_dir = Path(swm.data.utils.get_cache_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval_latent_pred_{run_id}.json"
    payload = {
        "ckpt_path": str(ckpt_path),
        "ctx_len": ctx_len,
        "history_size": history_size,
        "eval_horizons": eval_horizons,
        "eval_n_episodes_requested": eval_n_episodes,
        "eval_n_episodes_actual": int(rows[0]["n_episodes"]) if rows else 0,
        "eval_batch_size": eval_batch_size,
        "encoder_type": (cfg.get("encoder", {}).get("type", "vit")
                         if cfg.get("encoder") is not None else "vit"),
        "data_name": str(cfg.data.dataset.name),
        "results": rows,
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[eval_latent_prediction] wrote {out_path}")


if __name__ == "__main__":
    run()
