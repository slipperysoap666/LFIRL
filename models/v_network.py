from __future__ import annotations
from typing import Dict, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import numpy as np

# Reuse encoder and helpers to ensure identical behavior with Q.
from .q_network import (
    ObsEncoder,
    _ensure_tensor,
    _infer_device_from_obs,
)

class VNetwork(nn.Module):
    """
    Value network V(s) -> scalar.
    - Shares the same ObsEncoder as QNetwork to support dict/flat observations.
    - Provides set_obs_normalizer(...) to apply flat/per-key normalization.
    - Provides extreme_v_loss(...) implementing ExtremeV objective (XQL Eq.11).
    """

    def __init__(
        self,
        state_embed_dim: int,           # output dimension of ObsEncoder
        hidden: int = 256,
        layers: int = 3,
        act: str = "silu",
        layernorm: bool = False,
        obs_encoder: Optional[ObsEncoder] = None,
    ):
        super().__init__()
        self.obs_encoder = obs_encoder or ObsEncoder(out_dim=state_embed_dim)

        act_layer = nn.SiLU if str(act).lower() == "silu" else nn.ReLU

        mlp: list[nn.Module] = []
        last = state_embed_dim
        for _ in range(max(0, layers - 1)):
            mlp.append(nn.Linear(last, hidden))
            if layernorm:
                mlp.append(nn.LayerNorm(hidden))
            mlp.append(act_layer())
            last = hidden
        mlp.append(nn.Linear(last, 1))
        self.mlp = nn.Sequential(*mlp)

        # Optional normalizers (flat or per-key)
        self._flat_mean: Optional[torch.Tensor] = None
        self._flat_std: Optional[torch.Tensor] = None
        self._mean_dict: Dict[str, torch.Tensor] = {}
        self._std_dict: Dict[str, torch.Tensor] = {}

    def set_obs_normalizer(
        self,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        mean_dict: Optional[Dict[str, torch.Tensor]] = None,
        std_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Register normalization stats for obs (either flat or per-key)."""
        self._flat_mean = mean
        self._flat_std = std
        self._mean_dict = mean_dict or {}
        self._std_dict = std_dict or {}

    def _norm_obs(self, obs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply registered normalization to obs."""
        if isinstance(obs, torch.Tensor):
            if (self._flat_mean is None) or (self._flat_std is None):
                return obs
            if obs.dim() == 3:
                mean = self._flat_mean.view(1, 1, -1)
                std  = self._flat_std.view(1, 1, -1)
            else:
                mean = self._flat_mean.view(1, -1)
                std  = self._flat_std.view(1, -1)
            return (obs - mean) / (std + 1e-6)

        out: Dict[str, torch.Tensor] = {}
        for k, v in obs.items():
            m = self._mean_dict.get(k, None)
            s = self._std_dict.get(k, None)
            if (m is None) or (s is None):
                out[k] = v
            else:
                if v.dim() == 3:
                    out[k] = (v - m.view(1, 1, -1)) / (s.view(1, 1, -1) + 1e-6)
                else:
                    out[k] = (v - m.view(1, -1)) / (s.view(1, -1) + 1e-6)
        return out

    def forward(self, obs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Compute V(s). Accepts flat or dict obs (with/without time axis)."""
        device = _infer_device_from_obs(obs)
        if isinstance(obs, dict):
            obs = {k: _ensure_tensor(v, device) for k, v in obs.items()}
        else:
            obs = _ensure_tensor(obs, device)

        obs_n = self._norm_obs(obs)
        s_embed = self.obs_encoder(obs_n)   # [B, D]
        v = self.mlp(s_embed)               # [B, 1]
        return v.squeeze(-1)                # [B]

    @staticmethod
    def extreme_v_loss(
            v_values: torch.Tensor,
            q_hat: torch.Tensor,
            beta: float,
            reduction: str = "mean",
            detach_q: bool = True,
            clip_x: float = 50.0,  # add: numeric cap for stability
    ) -> torch.Tensor:
        if detach_q:
            q_hat = q_hat.detach()
        x = (q_hat - v_values) / float(beta)
        if clip_x is not None:
            x = torch.clamp(x, max=clip_x)  # avoid exp overflow
        # exp(x) - x - 1  ==  expm1(x) - x  (better near 0)
        loss = torch.expm1(x) - x
        return loss.mean() if reduction == "mean" else loss.sum()

    @staticmethod
    def _build_example_obs_with_time_from_env(env, obs_horizon: int, device: torch.device):
        """
        Construct a dummy observation with time axis To using env.reset():
          - flat: [1, To, S]
          - dict: {k: [1, To, Sk]}
        """
        obs, _ = env.reset()
        if isinstance(obs, dict):
            out = {}
            for k, v in obs.items():
                v = np.asarray(v, dtype=np.float32).reshape(1, 1, -1)  # [1,1,Sk]
                v = np.repeat(v, repeats=obs_horizon, axis=1)          # [1,To,Sk]
                out[k] = torch.tensor(v, dtype=torch.float32, device=device)
            return out
        else:
            v = np.asarray(obs, dtype=np.float32).reshape(1, 1, -1)    # [1,1,S]
            v = np.repeat(v, repeats=obs_horizon, axis=1)              # [1,To,S]
            return torch.tensor(v, dtype=torch.float32, device=device)

    @staticmethod
    def build_v_from_cfg(
        cfg: Dict[str, Any],
        example_obs: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        device: Optional[torch.device] = None,
        warmup_forward: bool = True,
    ) -> "VNetwork":
        """
        Build VNetwork from YAML-like dict.
        Accepts either {"v_net": {...}} or a pre-extracted v_cfg dict.
        If warmup_forward=True and example_obs is provided, run a dummy forward
        to materialize lazy per-key/flat MLPs on the correct device.
        """
        vcfg = cfg["v_net"] if ("v_net" in cfg and isinstance(cfg["v_net"], dict)) else cfg
        enc_cfg = vcfg.get("obs_encoder", {})

        obs_encoder = ObsEncoder(
            out_dim=int(enc_cfg.get("out_dim", 256)),
            per_key_hidden=int(enc_cfg.get("per_key_hidden", 256)),
            per_key_layers=int(enc_cfg.get("per_key_layers", 2)),
            agg=str(enc_cfg.get("agg", "sum")).lower(),
            concat_proj_dim=enc_cfg.get("concat_proj_dim", 256),
            act=str(enc_cfg.get("act", "silu")),
            layernorm=bool(enc_cfg.get("layernorm", False)),
            use_goal_delta=bool(enc_cfg.get("use_goal_delta", True)),
        )

        v = VNetwork(
            state_embed_dim=int(vcfg.get("state_embed_dim", obs_encoder.out_dim)),
            hidden=int(vcfg.get("hidden", 256)),
            layers=int(vcfg.get("layers", 3)),
            act=str(vcfg.get("act", "silu")),
            layernorm=bool(vcfg.get("layernorm", False)),
            obs_encoder=obs_encoder,
        )
        if device is not None:
            v = v.to(device)

        if warmup_forward and (example_obs is not None):
            v.eval()
            with torch.no_grad():
                dev = device or _infer_device_from_obs(example_obs)
                if isinstance(example_obs, dict):
                    dummy = {k: torch.zeros_like(val, device=dev) for k, val in example_obs.items()}
                else:
                    dummy = torch.zeros_like(example_obs, device=dev)
                _ = v(dummy)

        return v

    @staticmethod
    def build_v_from_cfg_with_env(
        cfg: Dict[str, Any],
        env,
        obs_horizon: int,
        device: Optional[torch.device] = None,
        warmup_forward: bool = True,
    ) -> "VNetwork":
        """
        Convenience builder: derive example_obs directly from env.reset()
        and then delegate to build_v_from_cfg(...).
        """
        device = device or torch.device("cpu")
        example_obs = VNetwork._build_example_obs_with_time_from_env(env, obs_horizon, device)
        return VNetwork.build_v_from_cfg(
            cfg=cfg,
            example_obs=example_obs,
            device=device,
            warmup_forward=warmup_forward,
        )
