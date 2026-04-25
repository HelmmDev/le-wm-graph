"""JEPA Implementation"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v

class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def _is_graph(self) -> bool:
        return getattr(self.encoder, "is_graph_encoder", False)

    def encode(self, info):
        """Encode observations and actions into embeddings.

        Two code paths, dispatched on `self.encoder.is_graph_encoder`:

        - Flat ViT path (default): runs encoder on (B*T, 3, H, W), pulls the
          CLS token, projects, and stores `info["emb"]` of shape (B, T, D).

        - Graph encoder path: runs encoder on (B*T, 3, H, W) PLUS GT
          `positions` (B*T, N_max, 2) and `mask` (B*T, N_max). Output tokens
          are (B*T, N_max, hidden_size). Projector is applied per-token then
          shape is restored to (B, T, N_max, D); the mask is broadcast back to
          (B, T, N_max) and stored in `info["emb_mask"]` so the predictor and
          downstream consumers can see which slots are real.

        Both paths populate `info["act_emb"]` if `info["action"]` is present.
        """
        pixels = info['pixels'].float()
        b = pixels.size(0)

        if self._is_graph():
            # Graph path: positions + mask flow into the encoder.
            assert "positions" in info, "graph encoder needs info['positions']"
            assert "mask" in info, "graph encoder needs info['mask']"
            positions = info["positions"].float()
            mask = info["mask"].bool()
            t = pixels.size(1)
            n_max = positions.size(2)

            pixels_flat = rearrange(pixels, "b t ... -> (b t) ...")
            positions_flat = rearrange(positions, "b t n d -> (b t) n d")
            mask_flat = rearrange(mask, "b t n -> (b t) n")

            output = self.encoder(
                pixels_flat,
                positions=positions_flat,
                mask=mask_flat,
                interpolate_pos_encoding=True,
            )
            tokens = output.last_hidden_state  # (B*T, N_max, H)

            # Per-token projector: flatten objects into the batch dim so the
            # projector (which expects (N, D)) can run unchanged.
            d_in = tokens.size(-1)
            emb_flat = self.projector(tokens.reshape(-1, d_in))
            d_out = emb_flat.size(-1)
            emb = emb_flat.view(b, t, n_max, d_out)
            info["emb"] = emb
            info["emb_mask"] = mask  # (B, T, N_max)
        else:
            pixels = rearrange(pixels, "b t ... -> (b t) ...")  # flatten for encoding
            output = self.encoder(pixels, interpolate_pos_encoding=True)
            pixels_emb = output.last_hidden_state[:, 0]  # cls token
            emb = self.projector(pixels_emb)
            info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb, mask=None):
        """Predict next state embedding.

        Flat ViT path:
            emb     : (B, T, D)
            act_emb : (B, T, A_emb)
            returns : (B, T, D)

        Graph path (auto-detected by emb.ndim == 4):
            emb     : (B, T, N_max, D)
            act_emb : (B, T, A_emb)
            mask    : (B, T, N_max) bool, REQUIRED.
            returns : (B, T, N_max, D)
        """
        if emb.ndim == 4:
            assert mask is not None, "graph predict() requires `mask`"
            preds = self.predictor(emb, act_emb, mask)  # (B, T, N, D)
            B, T, N, D = preds.shape
            preds = self.pred_proj(preds.reshape(-1, D)).view(B, T, N, -1)
            return preds

        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.

        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon

        For the graph encoder, simulator GT positions/mask are used at every
        rollout step (locked design decision; matches training).
        """

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, *_init["emb"].shape[1:])
        _init = {k: detach_clone(v) for k, v in _init.items()}

        graph = self._is_graph()
        if graph:
            # mask shape (B, T, N_max) -> (B, S, T, N_max)
            mask0 = _init["emb_mask"]
            mask = mask0.unsqueeze(1).expand(B, S, *mask0.shape[1:])
            mask = rearrange(mask, "b s ... -> (b s) ...").clone()

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # rollout predictor autoregressively for n_steps
        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]  # (BS, HS, ...)
            act_trunc = act_emb[:, -HS:]
            if graph:
                mask_trunc = mask[:, -HS:]
                pred_emb = self.predict(emb_trunc, act_trunc, mask=mask_trunc)[:, -1:]
                # The next slot's mask matches the latest known mask (GT
                # positions persist through the rollout — same N visible balls).
                mask = torch.cat([mask, mask[:, -1:]], dim=1)
            else:
                pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]
            emb = torch.cat([emb, pred_emb], dim=1)

            next_act = act_future[:, t : t + 1, :]
            act = torch.cat([act, next_act], dim=1)

        # predict the last state
        act_emb = self.action_encoder(act)
        emb_trunc = emb[:, -HS:]
        act_trunc = act_emb[:, -HS:]
        if graph:
            mask_trunc = mask[:, -HS:]
            pred_emb = self.predict(emb_trunc, act_trunc, mask=mask_trunc)[:, -1:]
        else:
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B,S, T-1, dim)
        goal_emb = info_dict["goal_emb"]  # (B, S, T, dim)

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given an info dict with goal and initial state."""

        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action")
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)

        return cost
