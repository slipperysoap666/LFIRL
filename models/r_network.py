from __future__ import annotations
from typing import Dict, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import numpy as np

# Reuse encoder and helpers to ensure identical behavior with Q/V.
from .q_network import (
    ObsEncoder,
    _ensure_tensor,
    _infer_device_from_obs,
    _action_shape_from_space,
)

class RewardNetwork(nn.Module):
    """
    Reward network r_theta(s, a) -> scalar (or r_theta(s) if include_action=False).

    Design goals:
      - Support both flat Tensor obs and Dict obs (with or without a time axis).
      - Share the same ObsEncoder as Q/V for consistent preprocessing.
      - Optional action input to cover r(s,a) vs r(s).
      - Optional input normalization for observations (flat or per-key).
      - Warm-up method that runs a dummy forward to materialize lazy modules.

    Notes:
      * For Discrete actions, pass a one-hot (or probability) vector of size n.
        This mirrors the convention used by QNetwork in this codebase.
    """

    def __init__(
        self,
        state_embed_dim: int,
        include_action: bool = True,
        action_dim: int = 0,
        hidden: int = 256,
        layers: int = 3,
        act: str = "silu",
        layernorm: bool = False,
        obs_encoder: Optional[ObsEncoder] = None,
    ):
        super().__init__()
        self.include_action = bool(include_action)
        self.action_dim = int(action_dim) if include_action else 0
        self.obs_encoder = obs_encoder or ObsEncoder(out_dim=state_embed_dim)

        act_layer = nn.SiLU if str(act).lower() == "silu" else nn.ReLU

        mlp: list[nn.Module] = []
        last = state_embed_dim + (self.action_dim if self.include_action else 0)
        for _ in range(max(0, layers - 1)):
            mlp.append(nn.Linear(last, hidden))
            if layernorm:
                mlp.append(nn.LayerNorm(hidden))
            mlp.append(act_layer())
            last = hidden
        mlp.append(nn.Linear(last, 1))
        self.mlp = nn.Sequential(*mlp)

        # Optional normalizers use the same API as Q/V.
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
        """Apply registered normalization to obs. Shapes may be [B,S] or [B,To,S]/[B,To,Sk]."""
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

    def forward(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute r_theta(s, a) or r_theta(s).

        Args:
            obs: Tensor [B,S] or [B,To,S], or Dict[str, Tensor] with shapes [B,Sk] or [B,To,Sk].
            action: Required if include_action=True. Shape [B,A] or [B,To,A] flattened internally.

        Returns:
            rewards: Tensor [B]
        """
        device = _infer_device_from_obs(obs)
        if isinstance(obs, dict):
            obs = {k: _ensure_tensor(v, device) for k, v in obs.items()}
        else:
            obs = _ensure_tensor(obs, device)

        obs_n = self._norm_obs(obs)
        s_embed = self.obs_encoder(obs_n)  # [B, D]

        if self.include_action:
            assert action is not None, "RewardNetwork.forward: 'action' is required when include_action=True."
            a = _ensure_tensor(action, device)
            if a.dim() >= 3:  # [B,To,A] -> [B, To*A]
                B = a.size(0)
                a = a.view(B, -1)
            x = torch.cat([s_embed, a], dim=-1)
        else:
            x = s_embed

        r = self.mlp(x)  # [B,1]
        return r.squeeze(-1)

    @staticmethod
    def build_example_obs_with_time_from_env(env, obs_horizon: int, device: torch.device):
        """
        Build an example observation with a time dimension To using env.reset():
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
    def build_reward_from_cfg(
        cfg: Dict[str, Any],
        example_obs: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        include_action: Optional[bool] = None,
        action_shape: Optional[Union[int, Tuple[int, ...]]] = None,
        device: Optional[torch.device] = None,
        warmup_forward: bool = True,
    ) -> "RewardNetwork":
        """
        Factory to build RewardNetwork from a YAML-like dict.
        Accepts either {"r_net": {...}} or a pre-extracted r_cfg dict.
        If warmup_forward=True and example_obs is provided, runs a dummy forward
        to materialize lazy modules on the right device.
        """
        rcfg = cfg["r_net"] if ("r_net" in cfg and isinstance(cfg["r_net"], dict)) else cfg
        enc_cfg = rcfg.get("obs_encoder", {})

        # Decide whether to include action
        if include_action is None:
            include_action = bool(rcfg.get("include_action", True))

        # Infer action dim (if needed)
        if include_action:
            if isinstance(action_shape, int):
                action_dim = action_shape
            else:
                assert action_shape is not None, "Action shape must be provided when include_action=True."
                action_dim = int(torch.tensor(list(action_shape)).prod().item())
        else:
            action_dim = 0

        # Build obs encoder
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

        rnet = RewardNetwork(
            state_embed_dim=int(rcfg.get("state_embed_dim", obs_encoder.out_dim)),
            include_action=include_action,
            action_dim=action_dim,
            hidden=int(rcfg.get("hidden", 256)),
            layers=int(rcfg.get("layers", 3)),
            act=str(rcfg.get("act", "silu")),
            layernorm=bool(rcfg.get("layernorm", False)),
            obs_encoder=obs_encoder,
        )
        if device is not None:
            rnet = rnet.to(device)

        # Optional warm-up forward to ensure modules are built on the correct device.
        if warmup_forward and (example_obs is not None):
            rnet.eval()
            with torch.no_grad():
                dev = device or _infer_device_from_obs(example_obs)
                if isinstance(example_obs, dict):
                    dummy_obs = {k: torch.zeros_like(v, device=dev) for k, v in example_obs.items()}
                    B = next(iter(dummy_obs.values())).size(0)
                else:
                    dummy_obs = torch.zeros_like(example_obs, device=dev)
                    B = dummy_obs.size(0)

                if include_action:
                    dummy_a = torch.zeros((B, action_dim), dtype=torch.float32, device=dev)
                else:
                    dummy_a = None
                _ = rnet(dummy_obs, dummy_a)

        return rnet

    @staticmethod
    def build_reward_from_cfg_with_env(
        cfg: Dict[str, Any],
        env,
        obs_horizon: int,
        device: Optional[torch.device] = None,
        warmup_forward: bool = True,
    ) -> "RewardNetwork":
        """
        Convenience builder: derive example_obs and (optionally) action_shape from env,
        then delegate to build_reward_from_cfg(...).
        """
        device = device or torch.device("cpu")
        example_obs = RewardNetwork.build_example_obs_with_time_from_env(env, obs_horizon, device)

        rcfg = cfg["r_net"] if ("r_net" in cfg and isinstance(cfg["r_net"], dict)) else cfg
        include_action = bool(rcfg.get("include_action", True))

        if include_action:
            act_shape = _action_shape_from_space(env.action_space)
        else:
            act_shape = None

        return RewardNetwork.build_reward_from_cfg(
            cfg=cfg,
            example_obs=example_obs,
            include_action=include_action,
            action_shape=act_shape,
            device=device,
            warmup_forward=warmup_forward,
        )
