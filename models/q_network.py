from __future__ import annotations

from typing import Dict, Optional, Tuple, Union, Any

import numpy as np
import torch
import torch.nn as nn


Tensor = torch.Tensor
ObsType = Union[Tensor, Dict[str, Tensor]]



def _maybe_flatten_time(x: torch.Tensor) -> torch.Tensor:
    """Accept [B, ...] or [B, To, ...]; if time dimension exists, flatten it into the feature dimension."""
    if x.dim() >= 3:
        B = x.size(0)
        rest = x.shape[1:]
        if len(rest) >= 2:
            To = rest[0]
            feat = int(torch.tensor(rest[1:]).prod().item())
            return x.reshape(B, To * feat)
    return x


def _ensure_tensor(x: Union[torch.Tensor, Any], device: torch.device) -> torch.Tensor:
    """Convert numpy/number to torch tensor; leave torch tensors unchanged."""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _add_goal_delta_if_possible(obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """If achieved_goal and desired_goal exist, add goal_delta = desired_goal - achieved_goal."""
    if ("achieved_goal" in obs_dict) and ("desired_goal" in obs_dict) and ("goal_delta" not in obs_dict):
        ag = obs_dict["achieved_goal"]
        dg = obs_dict["desired_goal"]
        obs_dict["goal_delta"] = dg - ag
    return obs_dict


def _infer_device_from_obs(obs: ObsType) -> torch.device:
    """Infer a reasonable device from obs containers."""
    if isinstance(obs, torch.Tensor):
        return obs.device
    for v in obs.values():
        if isinstance(v, torch.Tensor):
            return v.device
    return torch.device("cpu")


def _action_shape_from_space(space) -> Tuple[int, ...]:
    """Infer action shape from a Gymnasium/Gym space (Discrete or Box-like)."""
    if hasattr(space, "n") and not hasattr(space, "shape"):
        return (int(space.n),)
    if hasattr(space, "shape") and space.shape is not None:
        return tuple(int(x) for x in space.shape)
    raise ValueError("Unsupported action space type; expected Discrete or Box-like.")



class PerKeyMLP(nn.Module):
    """A small MLP used per dict key: R^{To*Sk or Sk} -> R^{D}."""
    def __init__(self, in_dim: int, out_dim: int, hidden: int, num_layers: int,
                 act: str = "silu", layernorm: bool = False):
        super().__init__()
        act_layer = nn.SiLU if act.lower() == "silu" else nn.ReLU

        layers = []
        last = in_dim
        for _ in range(max(0, num_layers - 1)):
            layers.append(nn.Linear(last, hidden))
            if layernorm:
                layers.append(nn.LayerNorm(hidden))
            layers.append(act_layer())
            last = hidden
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ObsEncoder(nn.Module):
    """
    Encode observations into a fixed-size vector:
      - Flat obs: Tensor [B, S] or [B, To, S]  --> flatten time if present --> MLP to [B, D]
      - Dict obs: {k: Tensor [B, Sk] or [B, To, Sk]} --> per-key MLP to [B, D], then aggreg. (sum/mean/concat+proj)
    """
    def __init__(
        self,
        out_dim: int = 256,
        per_key_hidden: int = 256,
        per_key_layers: int = 2,
        agg: str = "sum",               # 'sum' | 'mean' | 'concat'
        concat_proj_dim: Optional[int] = 256,
        act: str = "silu",
        layernorm: bool = False,
        use_goal_delta: bool = True,
    ):
        super().__init__()
        self.out_dim = int(out_dim)
        self.per_key_hidden = int(per_key_hidden)
        self.per_key_layers = int(per_key_layers)
        self.agg = str(agg).lower()
        self.concat_proj_dim = int(concat_proj_dim) if concat_proj_dim is not None else None
        self.act = act
        self.layernorm = bool(layernorm)
        self.use_goal_delta = bool(use_goal_delta)

        self._flat_mlp: Optional[PerKeyMLP] = None
        self._per_key_mlps = nn.ModuleDict()
        self._concat_proj: Optional[nn.Sequential] = None

    def _build_flat_mlp(self, in_dim: int, device: Optional[torch.device] = None) -> None:
        self._flat_mlp = PerKeyMLP(
            in_dim=in_dim, out_dim=self.out_dim,
            hidden=self.per_key_hidden, num_layers=self.per_key_layers,
            act=self.act, layernorm=self.layernorm
        )
        if device is not None:
            self._flat_mlp.to(device)

    def _build_key_mlp(self, key: str, in_dim: int, device: Optional[torch.device] = None) -> None:
        mlp = PerKeyMLP(
            in_dim=in_dim, out_dim=self.out_dim,
            hidden=self.per_key_hidden, num_layers=self.per_key_layers,
            act=self.act, layernorm=self.layernorm
        )
        if device is not None:
            mlp = mlp.to(device)
        self._per_key_mlps[key] = mlp

    def _build_concat_proj_if_needed(self, num_keys: int, device: Optional[torch.device] = None) -> None:
        if self.agg == "concat":
            proj_in = num_keys * self.out_dim
            proj_out = self.concat_proj_dim or self.out_dim
            proj = nn.Sequential(
                nn.Linear(proj_in, proj_out),
                nn.SiLU(),
                nn.Linear(proj_out, self.out_dim),
            )
            if device is not None:
                proj = proj.to(device)
            self._concat_proj = proj

    def forward(self, obs: ObsType) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            x = _maybe_flatten_time(obs)
            in_dim = x.shape[-1]
            if self._flat_mlp is None:
                self._build_flat_mlp(in_dim, device=x.device)
            return self._flat_mlp(x)

        obs_d = dict(obs)
        if self.use_goal_delta:
            obs_d = _add_goal_delta_if_possible(obs_d)

        feats = []
        keys = sorted(obs_d.keys())
        for k in keys:
            v = obs_d[k]
            v = _maybe_flatten_time(v)
            in_dim = v.shape[-1]
            if k not in self._per_key_mlps:
                self._build_key_mlp(k, in_dim, device=v.device)
            feats.append(self._per_key_mlps[k](v))

        if len(feats) == 0:
            dev = _infer_device_from_obs(obs_d)
            return torch.zeros((1, self.out_dim), device=dev, dtype=torch.float32)

        if self.agg == "sum":
            return torch.stack(feats, dim=0).sum(dim=0)
        if self.agg == "mean":
            return torch.stack(feats, dim=0).mean(dim=0)

        h = torch.cat(feats, dim=-1)
        if self._concat_proj is None:
            self._build_concat_proj_if_needed(num_keys=len(feats), device=h.device)
        return self._concat_proj(h)



class QNetwork(nn.Module):
    """
    Q network with a decomposed form:
        Q_full(s,a) = Q_hat(s,a) + b(s)

    - Q_hat(s,a): action-dependent part (trained in Stage1)
    - b(s): state-only bias/offset (trained in Stage1.5)

    forward(...) returns Q_full(s,a) if use_state_bias=True, else returns Q_hat(s,a).
    """
    def __init__(
        self,
        state_embed_dim: int,
        action_dim: int,
        hidden: int = 256,
        layers: int = 3,
        act: str = "silu",
        layernorm: bool = False,
        obs_encoder: Optional[ObsEncoder] = None,
        use_state_bias: bool = True,
        b_hidden: Optional[int] = None,
        b_layers: int = 2,
        b_layernorm: bool = False,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.obs_encoder = obs_encoder or ObsEncoder(out_dim=state_embed_dim)

        act_layer = nn.SiLU if act.lower() == "silu" else nn.ReLU

        # Q_hat head
        mlp = []
        last = state_embed_dim + action_dim
        for _ in range(max(0, layers - 1)):
            mlp.append(nn.Linear(last, hidden))
            if layernorm:
                mlp.append(nn.LayerNorm(hidden))
            mlp.append(act_layer())
            last = hidden
        mlp.append(nn.Linear(last, 1))
        self.mlp = nn.Sequential(*mlp)

        # b(s) head
        self.use_state_bias = bool(use_state_bias)
        self.b_mlp: Optional[nn.Sequential] = None
        if self.use_state_bias:
            bh = int(b_hidden) if b_hidden is not None else int(hidden)
            bl = int(b_layers)
            bnet = []
            last_b = state_embed_dim
            for _ in range(max(0, bl - 1)):
                bnet.append(nn.Linear(last_b, bh))
                if b_layernorm:
                    bnet.append(nn.LayerNorm(bh))
                bnet.append(act_layer())
                last_b = bh
            bnet.append(nn.Linear(last_b, 1))
            self.b_mlp = nn.Sequential(*bnet)

        # Optional input normalization
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
        self._flat_mean = mean
        self._flat_std = std
        self._mean_dict = mean_dict or {}
        self._std_dict = std_dict or {}

    def _norm_obs(self, obs: ObsType) -> ObsType:
        """Apply registered normalization. Shapes may be [B, S] or [B, To, S] / [B, To, Sk]."""
        if isinstance(obs, torch.Tensor):
            if (self._flat_mean is None) or (self._flat_std is None):
                return obs
            if obs.dim() == 3:
                mean = self._flat_mean.view(1, 1, -1)
                std = self._flat_std.view(1, 1, -1)
            else:
                mean = self._flat_mean.view(1, -1)
                std = self._flat_std.view(1, -1)
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

    def _encode_obs(self, obs: ObsType) -> torch.Tensor:
        device = _infer_device_from_obs(obs)
        if isinstance(obs, dict):
            obs = {k: _ensure_tensor(v, device) for k, v in obs.items()}
        else:
            obs = _ensure_tensor(obs, device)
        obs_n = self._norm_obs(obs)
        return self.obs_encoder(obs_n)  # [B, D]

    def _qhat_from_embed(self, s_embed: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        a_feat = _maybe_flatten_time(action)
        if a_feat.dim() != 2:
            a_feat = a_feat.view(a_feat.size(0), -1)
        x = torch.cat([s_embed, a_feat], dim=-1)
        q = self.mlp(x)
        return q.squeeze(-1)

    def q_hat_value(self, obs: ObsType, action: torch.Tensor) -> torch.Tensor:
        """Return Q_hat(s,a) (action-dependent only)."""
        device = _infer_device_from_obs(obs)
        action = _ensure_tensor(action, device)
        s_embed = self._encode_obs(obs)
        return self._qhat_from_embed(s_embed, action)

    def b_value(self, obs: ObsType) -> torch.Tensor:
        """Return b(s). If disabled, returns zeros."""
        s_embed = self._encode_obs(obs)
        if (not self.use_state_bias) or (self.b_mlp is None):
            return torch.zeros((s_embed.size(0),), device=s_embed.device, dtype=s_embed.dtype)
        return self.b_mlp(s_embed).squeeze(-1)

    def forward(self, obs: ObsType, action: torch.Tensor) -> torch.Tensor:
        """Return Q_full(s,a) = Q_hat(s,a) + b(s) if enabled."""
        device = _infer_device_from_obs(obs)
        action = _ensure_tensor(action, device)
        s_embed = self._encode_obs(obs)
        qhat = self._qhat_from_embed(s_embed, action)
        if (not self.use_state_bias) or (self.b_mlp is None):
            return qhat
        b = self.b_mlp(s_embed).squeeze(-1)
        return qhat + b

    def q_grad_wrt_action(
        self,
        obs: ObsType,
        action: torch.Tensor,
        retain_graph: bool = False,
        create_graph: bool = True,
    ) -> torch.Tensor:
        """Return dQ_full/da (same as dQ_hat/da). Ensures 'action' requires grad."""
        device = _infer_device_from_obs(obs)
        action_req = _ensure_tensor(action, device).clone().detach().requires_grad_(True)
        q = self.forward(obs, action_req)
        grads = torch.autograd.grad(
            outputs=q,
            inputs=action_req,
            grad_outputs=torch.ones_like(q),
            retain_graph=retain_graph,
            create_graph=create_graph,
            allow_unused=False,
        )[0]
        return grads

    def q_hat_parameters(self):
        """Parameters for Q_hat(s,a): encoder + Q mlp."""
        return list(self.obs_encoder.parameters()) + list(self.mlp.parameters())

    def b_parameters(self):
        """Parameters for b(s) head only."""
        if (not self.use_state_bias) or (self.b_mlp is None):
            return []
        return list(self.b_mlp.parameters())

    @staticmethod
    def build_q_from_cfg(
        cfg: Dict[str, Any],
        example_obs: Optional[ObsType] = None,
        action_shape: Union[int, Tuple[int, ...]] = None,
        device: Optional[torch.device] = None,
        warmup_forward: bool = True,
    ) -> "QNetwork":
        qcfg = cfg["q_net"] if ("q_net" in cfg and isinstance(cfg["q_net"], dict)) else cfg
        enc_cfg = qcfg.get("obs_encoder", {})

        if isinstance(action_shape, int):
            action_dim = int(action_shape)
        else:
            action_dim = int(torch.tensor(list(action_shape)).prod().item())

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

        use_state_bias = bool(qcfg.get("use_state_bias", True))
        b_hidden = qcfg.get("b_hidden", None)
        b_layers = int(qcfg.get("b_layers", 2))
        b_layernorm = bool(qcfg.get("b_layernorm", False))

        q = QNetwork(
            state_embed_dim=int(qcfg.get("state_embed_dim", obs_encoder.out_dim)),
            action_dim=action_dim,
            hidden=int(qcfg.get("hidden", 256)),
            layers=int(qcfg.get("layers", 3)),
            act=str(qcfg.get("act", "silu")),
            layernorm=bool(qcfg.get("layernorm", False)),
            obs_encoder=obs_encoder,
            use_state_bias=use_state_bias,
            b_hidden=b_hidden,
            b_layers=b_layers,
            b_layernorm=b_layernorm,
        )

        if device is not None:
            q = q.to(device)

        if warmup_forward and (example_obs is not None):
            q.eval()
            with torch.no_grad():
                dev = device or _infer_device_from_obs(example_obs)
                if isinstance(example_obs, dict):
                    dummy = {k: torch.zeros_like(v, device=dev) for k, v in example_obs.items()}
                    B = next(iter(dummy.values())).size(0)
                else:
                    dummy = torch.zeros_like(example_obs, device=dev)
                    B = dummy.size(0)
                dummy_a = torch.zeros((B, action_dim), dtype=torch.float32, device=dev)
                _ = q(dummy, dummy_a)

        return q

    @staticmethod
    def build_example_obs_with_time_from_env(env, obs_horizon: int, device: torch.device) -> ObsType:
        obs, _ = env.reset()
        if isinstance(obs, dict):
            out = {}
            for k, v in obs.items():
                v = np.asarray(v, dtype=np.float32).reshape(1, 1, -1)
                v = np.repeat(v, repeats=obs_horizon, axis=1)
                out[k] = torch.tensor(v, dtype=torch.float32, device=device)
            return out
        v = np.asarray(obs, dtype=np.float32).reshape(1, 1, -1)
        v = np.repeat(v, repeats=obs_horizon, axis=1)
        return torch.tensor(v, dtype=torch.float32, device=device)

    @staticmethod
    def build_q_from_cfg_with_env(
        cfg: Dict[str, Any],
        env,
        obs_horizon: int,
        device: Optional[torch.device] = None,
        warmup_forward: bool = True,
    ) -> "QNetwork":
        device = device or torch.device("cpu")
        example_obs = QNetwork.build_example_obs_with_time_from_env(env, obs_horizon, device)
        act_shape = _action_shape_from_space(env.action_space)
        return QNetwork.build_q_from_cfg(
            cfg=cfg,
            example_obs=example_obs,
            action_shape=act_shape,
            device=device,
            warmup_forward=warmup_forward,
        )

