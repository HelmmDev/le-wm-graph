import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack
from data.bouncing_balls_transform import BouncingBallsGraphTransform


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses.

    Loss masking (graph variant only):
        For the graph encoder, absent slots are filled by the encoder with the
        learned ``[ABSENT]`` embedding. We exclude those slots from the
        prediction MSE using ``output["emb_mask"]`` (shape (B, T, N_max), bool)
        so the model is not asked to "predict" a constant token. This keeps
        the matched-conditions comparison cleaner: with N_real=5 / N_max=8,
        ~37.5% of slots would otherwise contribute pure noise to the gradient
        and dilute the meaningful signal. The flat ViT path is preserved
        byte-identically (no mask available, plain ``.mean()`` reduction).
    """

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D) flat or (B, T, N, D) graph
    act_emb = output["act_emb"]
    emb_mask = output.get("emb_mask")  # (B, T, N) bool, only present for graph

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]
    ctx_mask = emb_mask[:, :ctx_len] if emb_mask is not None else None

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act, mask=ctx_mask) # pred

    # LeWM loss
    if emb.ndim == 4:
        # Graph variant: mask absent slots from pred_loss so the [ABSENT]
        # embedding doesn't contribute (constant target = gradient noise +
        # signal dilution). tgt_emb shape: (B, T_pred, N, D);
        # emb_mask shape: (B, T, N) — slice to match tgt_emb's time range.
        tgt_mask = emb_mask[:, n_preds:]  # (B, T_pred, N)
        diff_sq = (pred_emb - tgt_emb).pow(2)  # (B, T_pred, N, D)
        # Broadcast over the D dimension and average only over present slots.
        mask_expanded = tgt_mask.unsqueeze(-1).expand_as(diff_sq).float()
        output["pred_loss"] = (diff_sq * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)
    else:
        # Flat path: unchanged byte-identical behavior.
        output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    if emb.ndim == 4:
        # Graph variant: (B, T, N, D) → flatten N×D for byte-identical SIGReg
        # math (per project plan §SIGReg). transpose to (T, B, N*D).
        sigreg_input = emb.permute(1, 0, 2, 3).reshape(emb.size(1), emb.size(0), -1)
    else:
        sigreg_input = emb.transpose(0, 1)
    output["sigreg_loss"] = self.sigreg(sigreg_input)
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]

    # Optional dataset-specific transform group (see config/train/data/*.yaml).
    # Column normalizers run FIRST (on raw tensors with shapes matching the
    # HDF5 stats) and the dataset transform runs LAST, so any pad/mask/inject
    # operations don't have to worry about the normalizer's shape expectations.
    data_transform_cfg = cfg.data.get("transform", None)
    skip_normalize = set(
        (data_transform_cfg or {}).get("skip_normalize", []) or []
    )

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

            if col in skip_normalize:
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        # If the dataset has no `action` column (autonomous-physics envs like
        # bouncing balls), the dataset transform will inject a zero
        # placeholder of width `action_dim`. Mirror that here so the predictor
        # builds the right-sized action_encoder.
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

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    ##############################
    ##       model / optim      ##
    ##############################

    # Encoder dispatch. The default (no `encoder` group, or encoder.type=='vit')
    # preserves the original flat ViT path byte-identically. encoder.type='graph'
    # selects the LeWM graph encoder built by Agent F (models/graph_encoder.py).
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
        from models import GraphEncoder
        encoder = GraphEncoder(
            image_size=cfg.img_size,
            d_obj=encoder_cfg.d_obj,
            n_max=encoder_cfg.n_max,
            capacity=encoder_cfg.capacity,
            width_mode=encoder_cfg.width_mode,
        )
        # Per-token embed_dim. The graph encoder produces (B*T, N_max, d_obj)
        # and the SetPredictor reshapes to (B, T*N_max, d_obj) with attention
        # mask. Both `native` and `projected` width_modes operate per-token —
        # `projected` adds an internal Linear(d_obj, 192) before output, so its
        # per-token width is fixed at 192; `native` keeps per-token = d_obj.
        # Width "widening" (per the ablation) only differs from baseline when
        # d_obj > 192 in `native` mode.
        if encoder_cfg.width_mode == "projected":
            embed_dim = 192
        elif encoder_cfg.width_mode == "native":
            embed_dim = int(encoder_cfg.d_obj)
        else:
            raise ValueError(
                f"Unknown encoder.width_mode={encoder_cfg.width_mode!r}; "
                "expected 'native' or 'projected'."
            )
        with open_dict(cfg):
            cfg.wm.embed_dim = embed_dim
        # GraphEncoder produces predictor-ready embeddings; downstream MLPs
        # below still expect a `hidden_dim`. Use embed_dim so the projector /
        # predictor MLPs are dimension-preserving on the encoder output.
        hidden_dim = embed_dim
    else:
        raise ValueError(
            f"Unknown encoder.type={encoder_type!r}; expected 'vit' or 'graph'."
        )

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

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()
    return


if __name__ == "__main__":
    run()
