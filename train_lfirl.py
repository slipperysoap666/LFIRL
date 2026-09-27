from __future__ import annotations

import os
import sys
import argparse
import yaml
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from tqdm import tqdm
from typing import Optional, Dict, Any, Tuple, Union, List

from utils.utils import (
    make_env,
    evaluate_policy_receding,
    compute_obs_stats_flat,
    compute_obs_stats_dict,
    _collate_obs_list,
    canonicalize_obs,
)
from utils.preprocess import (
    make_action_tensor,
    windowize_obs,
    get_traj_slices_prefer_offline,
)
from utils.logger import Logger
from utils.evaluation import evaluate_reward_via_sac

from policy_training.train_diffusion_policy import (
    train_diffusion_policy,
    build_diffusion_policy_model,
    get_policy_checkpoint_candidates,
    resolve_policy_backend,
)
from models.diffusion_model import DiffusionPolicyTimeseries
from models.q_network import QNetwork
from models.v_network import VNetwork
from models.r_network import RewardNetwork

from datasets.dataset import OfflineDataset

Tensor = torch.Tensor
ObsType = Union[Tensor, Dict[str, Tensor]]


def sanitize_env_id(env_id: str) -> str:
    return env_id.replace("/", "_").replace(":", "_")


def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_sb3_policy_name(observation_space: gym.Space, requested_policy: Optional[str]) -> str:
    policy = str(requested_policy).strip() if requested_policy is not None else "auto"
    if policy and policy.lower() != "auto":
        return policy
    return "MultiInputPolicy" if isinstance(observation_space, gym.spaces.Dict) else "MlpPolicy"


def _format_action_for_env_step(action: Any, action_space: gym.Space):
    if isinstance(action_space, gym.spaces.Discrete):
        return int(np.asarray(action).item())

    arr = np.asarray(action, dtype=np.float32)
    shape = getattr(action_space, "shape", None)
    if shape is not None:
        arr = arr.reshape(shape)
    return arr


def _format_action_for_storage(action: Any, action_space: gym.Space) -> np.ndarray:
    if isinstance(action_space, gym.spaces.Discrete):
        idx = int(np.asarray(action).item())
        vec = np.zeros((action_space.n,), dtype=np.float32)
        vec[idx] = 1.0
        return vec

    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    return arr.astype(np.float32, copy=False)


def _stack_canonical_obs_records(
    obs_records: List[Union[np.ndarray, Dict[str, np.ndarray]]]
) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
    if len(obs_records) == 0:
        raise RuntimeError("Cannot stack empty observation records.")

    first = obs_records[0]
    if isinstance(first, dict):
        keys = sorted(first.keys())
        states_dict = {
            k: np.stack(
                [np.asarray(obs[k], dtype=np.float32).reshape(-1) for obs in obs_records],
                axis=0,
            ).astype(np.float32)
            for k in keys
        }
        states = np.concatenate([states_dict[k] for k in keys], axis=1).astype(np.float32)
        return states, states_dict

    states = np.stack(
        [np.asarray(obs, dtype=np.float32).reshape(-1) for obs in obs_records],
        axis=0,
    ).astype(np.float32)
    return states, None


def _count_transitions_from_slices(traj_slices: List[Tuple[int, int]]) -> int:
    total = 0
    for start, end_exclusive in traj_slices:
        total += max(0, int(end_exclusive) - int(start) - 1)
    return total


def _count_batches(total_transitions: int, batch_size: int) -> int:
    if total_transitions <= 0:
        return 0
    return (int(total_transitions) + int(batch_size) - 1) // int(batch_size)


def collect_online_rollout_source(
    *,
    env_id: str,
    online_cfg: dict,
    num_trajectories: int,
    base_seed: int,
) -> Dict[str, Any]:
    algo = str(online_cfg.get("algorithm", "ppo")).lower()
    if algo != "ppo":
        raise ValueError(f"Unsupported online.algorithm='{algo}'. Expected 'ppo'.")

    try:
        from stable_baselines3 import PPO
    except Exception as e:
        raise RuntimeError(
            "Online V/R rollout collection requires stable-baselines3 PPO, "
            f"but the import failed under Python '{sys.executable}': {e}"
        )

    if num_trajectories <= 0:
        raise RuntimeError("online.num_trajectories must be > 0.")

    seed_offset = int(online_cfg.get("seed_offset", 100000))
    rollout_seed = int(online_cfg.get("seed", base_seed + seed_offset))
    deterministic = bool(online_cfg.get("deterministic", False))
    requested_device = str(online_cfg.get("device", "auto")).lower()
    sb3_device = requested_device if requested_device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    rollout_env = make_env(env_id, seed=rollout_seed)
    try:
        sb3_policy = _resolve_sb3_policy_name(
            rollout_env.observation_space,
            online_cfg.get("policy", "auto"),
        )

        ppo_kwargs: Dict[str, Any] = dict(
            policy=sb3_policy,
            env=rollout_env,
            verbose=int(online_cfg.get("verbose", 0)),
            seed=rollout_seed,
            device=sb3_device,
            n_steps=max(int(online_cfg.get("n_steps", 2048)), 2),
            batch_size=max(int(online_cfg.get("batch_size", 64)), 2),
            learning_rate=float(online_cfg.get("learning_rate", 3e-4)),
            gamma=float(online_cfg.get("gamma", 0.99)),
            n_epochs=max(int(online_cfg.get("n_epochs", 10)), 1),
        )
        policy_kwargs = online_cfg.get("policy_kwargs", None)
        if isinstance(policy_kwargs, dict) and len(policy_kwargs) > 0:
            ppo_kwargs["policy_kwargs"] = policy_kwargs

        model = PPO(**ppo_kwargs)

        obs_records: List[Union[np.ndarray, Dict[str, np.ndarray]]] = []
        action_records: List[np.ndarray] = []
        done_records: List[bool] = []
        traj_slices: List[Tuple[int, int]] = []
        episode_lengths: List[int] = []
        cursor = 0

        print(
            f"[Online] Collecting {num_trajectories} trajectories using initialized PPO "
            f"(policy={sb3_policy}, deterministic={deterministic}, device={sb3_device})."
        )

        for ep in range(num_trajectories):
            obs_raw, _info = rollout_env.reset(seed=rollout_seed + ep)
            ep_obs: List[Union[np.ndarray, Dict[str, np.ndarray]]] = []
            ep_actions: List[np.ndarray] = []
            ep_dones: List[bool] = []
            done = False

            while not done:
                ep_obs.append(canonicalize_obs(obs_raw))

                action_pred, _state = model.predict(obs_raw, deterministic=deterministic)
                env_action = _format_action_for_env_step(action_pred, rollout_env.action_space)
                stored_action = _format_action_for_storage(env_action, rollout_env.action_space)

                obs_raw, _reward, terminated, truncated, _info = rollout_env.step(env_action)
                done = bool(terminated or truncated)

                ep_actions.append(stored_action)
                ep_dones.append(done)

            if len(ep_actions) == 0:
                continue

            obs_records.extend(ep_obs)
            action_records.extend(ep_actions)
            done_records.extend(ep_dones)
            traj_slices.append((cursor, cursor + len(ep_actions)))
            episode_lengths.append(len(ep_actions))
            cursor += len(ep_actions)

        del model

    finally:
        rollout_env.close()

    if cursor <= 0 or len(traj_slices) == 0:
        raise RuntimeError("Online PPO rollout collection produced no usable trajectories.")

    states, states_dict = _stack_canonical_obs_records(obs_records)
    actions = np.stack(action_records, axis=0).astype(np.float32)
    dones = np.asarray(done_records, dtype=np.bool_)

    mean_len = float(np.mean(episode_lengths)) if episode_lengths else 0.0
    print(
        f"[Online] Ready: episodes={len(traj_slices)} | steps={cursor} | "
        f"mean_ep_len={mean_len:.1f}"
    )

    return dict(
        name="online_ppo_init",
        states_src=states_dict if states_dict is not None else states,
        states=states,
        states_dict=states_dict,
        actions=actions,
        dones=dones,
        traj_slices=traj_slices,
    )


@torch.no_grad()
def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float = 0.005):
    """Polyak / EMA update: target <- (1-tau)*target + tau*source."""
    for p_t, p in zip(target.parameters(), source.parameters()):
        p_t.data.lerp_(p.data, tau)
    for b_t, b in zip(target.buffers(), source.buffers()):
        b_t.data.copy_(b.data)


class RunningMeanStd:
    """Track mean/std online to standardize target_r during training."""

    def __init__(self, eps: float = 1e-4):
        self.count = eps
        self.mean = 0.0
        self.var = 1.0

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.detach()
        batch_mean = x.mean().item()
        batch_var = x.var(unbiased=False).item()
        batch_count = float(x.numel())

        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / tot)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta * delta * (self.count * batch_count / tot)
        new_var = M2 / tot

        self.mean = new_mean
        self.var = max(new_var, 1e-6)
        self.count = tot

    @property
    def std(self):
        return math.sqrt(self.var)


def _build_flat_obs_for_policy(
    obs: ObsType,
    policy_obs_keys: Optional[List[str]],
    device: torch.device,
) -> torch.Tensor:
    """
    Convert dict-obs {k:[B,To,Sk]} to flat obs [B,To,S] using policy_obs_keys order.
    If obs is already Tensor -> return [B,To,S] tensor.
    """
    if isinstance(obs, torch.Tensor):
        x = obs.to(device=device, dtype=torch.float32)
        if x.dim() == 2:
            # [To,S] -> [1,To,S]
            x = x.unsqueeze(0)
        return x

    assert policy_obs_keys is not None and len(policy_obs_keys) > 0, \
        "policy_obs_keys must be provided for dict-obs flattening."

    parts = []
    for k in policy_obs_keys:
        v = obs[k]
        if not isinstance(v, torch.Tensor):
            v = torch.as_tensor(v, dtype=torch.float32)
        if v.dim() == 2:
            v = v.unsqueeze(0)  # [1,To,Sk]
        parts.append(v.to(device=device, dtype=torch.float32))
    return torch.cat(parts, dim=-1)  # [B,To,S]


def _sample_indices(indices: np.ndarray, max_samples: int) -> np.ndarray:
    if max_samples is None or max_samples <= 0 or len(indices) <= max_samples:
        return indices
    return np.random.choice(indices, size=max_samples, replace=False)


@torch.no_grad()
def warmup_fit_dp_normalizer(
    dp_model: DiffusionPolicyTimeseries,
    offline: OfflineDataset,
    policy_obs_keys: Optional[List[str]],
    device: torch.device,
    mode: str = "gaussian",          # MUST be "gaussian" or "limits"
    max_samples: int = 200000,
):
    """
    Warm-up fit to create normalizer internal params/buffers (structure).
    Immediately after this, we will load ckpt again to overwrite with true stats.
    """
    if not hasattr(dp_model, "normalizer") or dp_model.normalizer is None:
        print("[DP-Norm] No normalizer on dp_model; skip warmup fit.")
        return

    if mode not in ["gaussian", "limits"]:
        mode = "gaussian"

    if hasattr(offline, "train_indices") and offline.train_indices is not None:
        idx = np.asarray(offline.train_indices, dtype=np.int64)
    else:
        idx = np.arange(len(offline.actions), dtype=np.int64)
    idx = _sample_indices(idx, max_samples)

    if getattr(offline, "states_dict", None) is not None:
        assert policy_obs_keys is not None and len(policy_obs_keys) > 0
        obs_arr = np.concatenate([offline.states_dict[k][idx] for k in policy_obs_keys], axis=-1).astype(np.float32)
    else:
        obs_arr = np.asarray(offline.states[idx], dtype=np.float32)

    act_arr = np.asarray(offline.actions[idx], dtype=np.float32)

    if obs_arr.ndim == 1:
        obs_arr = obs_arr.reshape(-1, 1)
    if act_arr.ndim == 1:
        act_arr = act_arr.reshape(-1, 1)

    payload = {
        "obs": torch.as_tensor(obs_arr, dtype=torch.float32, device=device),
        "action": torch.as_tensor(act_arr, dtype=torch.float32, device=device),
    }

    # Some implementations want seq input; we try flat first, then seq.
    try:
        dp_model.normalizer.fit(payload, mode=mode)
        print(f"[DP-Norm] Warmup fit OK (flat). mode={mode}, N={len(idx)}")
        return
    except Exception as e1:
        payload2 = {
            "obs": payload["obs"].unsqueeze(1),      # [N,1,S]
            "action": payload["action"].unsqueeze(1) # [N,1,A]
        }
        dp_model.normalizer.fit(payload2, mode=mode)
        print(f"[DP-Norm] Warmup fit OK (seq). mode={mode}, N={len(idx)}")
        return


def load_dp_with_normalizer_preload(
    *,
    dp_model: DiffusionPolicyTimeseries,
    ckpt_path: str,
    offline: OfflineDataset,
    policy_obs_keys: Optional[List[str]],
    device: torch.device,
    mode: str = "gaussian",
    max_samples: int = 200000,
) -> Tuple[List[str], List[str]]:
    raw_state = torch.load(ckpt_path, map_location=device)
    missing, unexpected = dp_model.load_state_dict(raw_state, strict=False)

    # If normalizer params appear in ckpt but are unexpected, it means normalizer structure not created yet.
    has_norm_unexpected = any(k.startswith("normalizer.") for k in unexpected)
    if has_norm_unexpected:
        print(f"[DP-Norm] Found {sum(k.startswith('normalizer.') for k in unexpected)} unexpected normalizer keys. "
              f"Warmup-fit to create structure, then reload ckpt to preload.")
        warmup_fit_dp_normalizer(
            dp_model=dp_model,
            offline=offline,
            policy_obs_keys=policy_obs_keys,
            device=device,
            mode=mode,
            max_samples=max_samples,
        )
        # Reload again: now normalizer keys should be loadable
        missing, unexpected = dp_model.load_state_dict(raw_state, strict=False)

    return missing, unexpected


@torch.no_grad()
def dp_action_score_from_policy(
    dp_model: DiffusionPolicyTimeseries,
    obs_t: ObsType,
    action_t: torch.Tensor,  # [B,A] or [A]
    device: torch.device,
    policy_obs_keys: Optional[List[str]],
    t_eval: Optional[int] = None,
) -> torch.Tensor:
    """
    Use pretrained diffusion policy to get score at (s,a):
      g*(s,a) ~= d/da log p(a|s) in original action space.

    IMPORTANT:
      - If env obs is dict, dp policy was trained on flattened obs_dim. We must flatten here too.
    """
    dp_model.eval()

    # obs -> [B,To,S]
    obs_batch = _build_flat_obs_for_policy(obs_t, policy_obs_keys, device=device)

    # action -> [B,A]
    if not isinstance(action_t, torch.Tensor):
        action_b = torch.as_tensor(action_t, dtype=torch.float32, device=device)
    else:
        action_b = action_t.to(device=device, dtype=torch.float32)
    if action_b.dim() == 1:
        action_b = action_b.unsqueeze(0)

    B, A = action_b.shape
    Tp = int(dp_model.n_action_steps)
    actions_seq = action_b.unsqueeze(1).repeat(1, Tp, 1)  # [B,Tp,A]

    g_star = dp_model.score_first_action(
        obs=obs_batch,
        action_seq=actions_seq,
        t_eval=t_eval,
    )  # [B,A]

    return g_star.contiguous()


def _qhat_value(q_hat: QNetwork, obs: ObsType, action: torch.Tensor) -> torch.Tensor:
    """Return the action-dependent Qhat term, avoiding any internal state-bias head."""
    if hasattr(q_hat, "q_hat_value"):
        return q_hat.q_hat_value(obs, action)
    return q_hat(obs, action)


def _select_obs_by_mask(obs: ObsType, mask: torch.Tensor) -> ObsType:
    if isinstance(obs, dict):
        return {k: v[mask] for k, v in obs.items()}
    return obs[mask]


def _repeat_obs_for_samples(obs: ObsType, repeats: int) -> ObsType:
    if repeats <= 1:
        return obs
    if isinstance(obs, dict):
        return {
            k: v.repeat(repeats, *([1] * (v.dim() - 1)))
            for k, v in obs.items()
        }
    return obs.repeat(repeats, *([1] * (obs.dim() - 1)))


def _sample_uniform_actions(
    *,
    num_actions: int,
    action_dim: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    low = torch.as_tensor(action_low, dtype=torch.float32, device=device).view(1, action_dim)
    high = torch.as_tensor(action_high, dtype=torch.float32, device=device).view(1, action_dim)
    return low + (high - low) * torch.rand((num_actions, action_dim), device=device)


def train_qhat_LFIRL_step(
    *,
    q_hat: QNetwork,
    q_optim: torch.optim.Optimizer,
    dp_model: DiffusionPolicyTimeseries,
    obs: ObsType,
    action: torch.Tensor,
    device: torch.device,
    policy_obs_keys: Optional[List[str]],
    q_loss_cfg: dict,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> float:
    """
    Stage1:
      - LFIRL gradient matching: || dQhat/da - score_dp ||^2
      - Ranking loss: encourage Qhat(s,a_expert) > Qhat(s,a_rand) + margin
    """
    q_hat.train()

    beta_scale = float(q_loss_cfg.get("beta_scale", 1.0))
    use_huber = bool(q_loss_cfg.get("use_huber", False))
    huber_delta = float(q_loss_cfg.get("huber_delta", 1.0))
    clip_grad_norm = float(q_loss_cfg.get("clip_grad_norm", 1.0))
    q_l2_reg = float(q_loss_cfg.get("q_l2_reg", 0.0))

    rank_weight = float(q_loss_cfg.get("rank_weight", 0.0))
    rank_margin = float(q_loss_cfg.get("rank_margin", 1.0))
    rank_samples = int(q_loss_cfg.get("rank_samples", 4))
    dp_t_eval_cfg = q_loss_cfg.get("dp_t_eval", None)
    dp_t_eval = None if dp_t_eval_cfg is None else int(dp_t_eval_cfg)
    if dp_t_eval is not None and dp_t_eval < 0:
        dp_t_eval = None

    with torch.no_grad():
        g_star = dp_action_score_from_policy(
            dp_model=dp_model,
            obs_t=obs,
            action_t=action,
            device=device,
            policy_obs_keys=policy_obs_keys,
            t_eval=dp_t_eval,
        )
        g_star = beta_scale * g_star  # [B,A]

    g_hat = q_hat.q_grad_wrt_action(
        obs=obs,
        action=action,
        retain_graph=True,
        create_graph=True,
    )
    if g_hat.dim() > 2:
        g_hat = g_hat.view(g_hat.size(0), -1)

    if use_huber:
        grad_loss = F.smooth_l1_loss(g_hat, g_star, beta=huber_delta)
    else:
        grad_loss = F.mse_loss(g_hat, g_star)

    loss = grad_loss

    if rank_weight > 0.0 and rank_samples > 0:
        B = action.size(0)
        A = action.size(1)
        low = torch.as_tensor(action_low, dtype=torch.float32, device=device).view(1, A)
        high = torch.as_tensor(action_high, dtype=torch.float32, device=device).view(1, A)

        rand_actions = low + (high - low) * torch.rand((B * rank_samples, A), device=device)
        if isinstance(obs, dict):
            obs_rep = {k: v.repeat(rank_samples, 1, 1) for k, v in obs.items()}
        else:
            obs_rep = obs.repeat(rank_samples, 1, 1)

        q_expert = _qhat_value(q_hat, obs, action).view(-1)                         # [B]
        q_rand = _qhat_value(q_hat, obs_rep, rand_actions).view(rank_samples, B).t() # [B,K]

        # hinge: margin - (q_e - q_r)
        hinge = F.relu(rank_margin - (q_expert.unsqueeze(1) - q_rand))
        rank_loss = hinge.mean()
        loss = loss + rank_weight * rank_loss

    if q_l2_reg > 0.0:
        q_vals = _qhat_value(q_hat, obs, action).view(-1)
        loss = loss + q_l2_reg * (q_vals ** 2).mean()

    q_optim.zero_grad(set_to_none=True)
    loss.backward()
    if clip_grad_norm and clip_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(q_hat.parameters(), clip_grad_norm)
    q_optim.step()

    return float(loss.item())


def _backup_value_from_qhat(
    *,
    q_hat: QNetwork,
    b_net: VNetwork,
    next_obs: ObsType,
    action_dim: int,
    device: torch.device,
    action_low: np.ndarray,
    action_high: np.ndarray,
    num_samples: int,
    backup_type: str = "max",
    lse_beta: float = 1.0,
) -> torch.Tensor:
    """
      V(next_s) ~= b(next_s) + max_a Qhat(next_s,a)
      V(next_s) ~= b(next_s) + max_a Qhat(next_s,a)
      or b(next_s) + beta * logsumexp(Qhat/beta)
    """
    if num_samples <= 0:
        # degenerate: just b(s')
        return b_net(next_obs).view(-1)

    B = next_obs["observation"].size(0) if isinstance(next_obs, dict) else next_obs.size(0)
    A = action_dim

    low = torch.as_tensor(action_low, dtype=torch.float32, device=device).view(1, A)
    high = torch.as_tensor(action_high, dtype=torch.float32, device=device).view(1, A)
    a_rand = low + (high - low) * torch.rand((B * num_samples, A), device=device)

    if isinstance(next_obs, dict):
        obs_rep = {k: v.repeat(num_samples, 1, 1) for k, v in next_obs.items()}
    else:
        obs_rep = next_obs.repeat(num_samples, 1, 1)

    with torch.no_grad():
        qn = q_hat(obs_rep, a_rand).view(num_samples, B).t()  # [B,K]

    b_next = b_net(next_obs).view(-1)  # [B]

    if backup_type.lower() == "logsumexp":
        beta = max(float(lse_beta), 1e-6)
        v = beta * torch.logsumexp(qn / beta, dim=1) - beta * math.log(float(num_samples))
        return b_next + v
    else:
        return b_next + torch.max(qn, dim=1).values


def train_b_scale_step(
    *,
    q_hat: QNetwork,
    b_net: VNetwork,
    b_optim: torch.optim.Optimizer,
    r_net: Optional[RewardNetwork],
    r_optim: Optional[torch.optim.Optimizer],
    obs: ObsType,
    action: torch.Tensor,
    next_obs: ObsType,
    done_t: torch.Tensor,
    gamma: float,
    device: torch.device,
    action_dim: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    cfg: dict,
) -> float:
    """
    Legacy helper (not used by the LFIRL main loop):
      Freeze Qhat.
      Learn b(s) using Bellman residual on Qfull = Qhat + b(s).
      Optionally also update r_net here (fit_reward=true).
    """
    b_net.train()
    if r_net is not None:
        r_net.train()

    loss_cfg = cfg.get("loss", {}) if isinstance(cfg, dict) else {}
    fit_reward = bool(loss_cfg.get("fit_reward", False))
    num_next_samples = int(loss_cfg.get("num_next_action_samples", 16))
    backup_type = str(loss_cfg.get("backup_type", "max")).lower()
    lse_beta = float(loss_cfg.get("lse_beta", 1.0))

    use_huber = bool(loss_cfg.get("use_huber", True))
    huber_delta = float(loss_cfg.get("huber_delta", 1.0))
    clip_grad_norm = float(loss_cfg.get("clip_grad_norm", 1.0))

    with torch.no_grad():
        q_sa = q_hat(obs, action).view(-1)

    b_s = b_net(obs).view(-1)
    q_full_sa = q_sa + b_s

    if fit_reward and (r_net is not None):
        r_pred = r_net(obs, action).view(-1)
    else:
        r_pred = torch.zeros_like(q_full_sa)

    v_next = _backup_value_from_qhat(
        q_hat=q_hat,
        b_net=b_net,
        next_obs=next_obs,
        action_dim=action_dim,
        device=device,
        action_low=action_low,
        action_high=action_high,
        num_samples=num_next_samples,
        backup_type=backup_type,
        lse_beta=lse_beta,
    )

    target = r_pred + float(gamma) * (1.0 - done_t) * v_next
    td = q_full_sa - target

    if use_huber:
        loss = F.smooth_l1_loss(td, torch.zeros_like(td), beta=huber_delta)
    else:
        loss = (td ** 2).mean()

    b_optim.zero_grad(set_to_none=True)
    if fit_reward and (r_optim is not None):
        r_optim.zero_grad(set_to_none=True)

    loss.backward()

    if clip_grad_norm and clip_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(b_net.parameters(), clip_grad_norm)
        if fit_reward and (r_net is not None):
            torch.nn.utils.clip_grad_norm_(r_net.parameters(), clip_grad_norm)

    b_optim.step()
    if fit_reward and (r_optim is not None):
        r_optim.step()

    return float(loss.item())


def train_b_offset_step(
    *,
    q_hat: QNetwork,
    v_net: VNetwork,
    b_net: VNetwork,
    b_optim: torch.optim.Optimizer,
    obs: ObsType,
    action: torch.Tensor,
    next_obs: ObsType,
    done_t: torch.Tensor,
    gamma: float,
    device: torch.device,
    action_dim: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    cfg: dict,
) -> float:
    """
    Stage II in LFIRL:
      freeze Qhat and the initial V, then learn the state-only offset b(s)
      with the Bellman-style calibration and terminal anchor from Eq. (11).
    """
    q_hat.eval()
    v_net.eval()
    b_net.train()

    loss_cfg = cfg.get("loss", {}) if isinstance(cfg, dict) else {}
    use_huber = bool(loss_cfg.get("use_huber", True))
    huber_delta = float(loss_cfg.get("huber_delta", 1.0))
    clip_grad_norm = float(loss_cfg.get("clip_grad_norm", 1.0))

    offset_l2_reg = float(
        loss_cfg.get(
            "offset_l2_reg",
            loss_cfg.get("b_l2_reg", loss_cfg.get("lambda_b", 0.0)),
        )
    )
    terminal_anchor_weight = float(loss_cfg.get("terminal_anchor_weight", 1.0))
    terminal_action_samples = max(int(loss_cfg.get("terminal_action_samples", 1)), 1)

    done_f = done_t.to(device=device, dtype=torch.float32).view(-1)

    with torch.no_grad():
        q_sa = _qhat_value(q_hat, obs, action).view(-1)
        v_next = v_net(next_obs).view(-1)

    b_s = b_net(obs).view(-1)
    b_next = b_net(next_obs).view(-1)

    next_value = (v_next + b_next) * (1.0 - done_f)
    delta = q_sa + b_s - float(gamma) * next_value

    if use_huber:
        transition_loss = F.smooth_l1_loss(delta, torch.zeros_like(delta), beta=huber_delta)
    else:
        transition_loss = delta.pow(2).mean()

    loss = transition_loss

    if offset_l2_reg > 0.0:
        loss = loss + offset_l2_reg * b_s.pow(2).mean()

    terminal_mask = done_f > 0.5
    if terminal_anchor_weight > 0.0 and bool(terminal_mask.any().item()):
        term_obs = _select_obs_by_mask(next_obs, terminal_mask)
        term_count = int(terminal_mask.sum().item())
        term_obs_rep = _repeat_obs_for_samples(term_obs, terminal_action_samples)
        term_actions = _sample_uniform_actions(
            num_actions=term_count * terminal_action_samples,
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
            device=device,
        )
        with torch.no_grad():
            q_term = _qhat_value(q_hat, term_obs_rep, term_actions).view(-1)
        b_term = b_net(term_obs_rep).view(-1)
        terminal_loss = (q_term + b_term).pow(2).mean()
        loss = loss + terminal_anchor_weight * terminal_loss

    b_optim.zero_grad(set_to_none=True)
    loss.backward()
    if clip_grad_norm and clip_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(b_net.parameters(), clip_grad_norm)
    b_optim.step()

    return float(loss.item())


class QHatOnlyWrapper(nn.Module):
    """Frozen view of the uncalibrated Qhat used for initial Gumbel value fitting."""
    def __init__(self, q_hat: QNetwork):
        super().__init__()
        self.q_hat = q_hat

    def forward(self, obs: ObsType, action: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return _qhat_value(self.q_hat, obs, action).view(-1)

    def q_value(self, obs: ObsType, action: torch.Tensor) -> torch.Tensor:
        return self.forward(obs, action)


class QFullWrapper(nn.Module):
    """
    Q_full(s,a) = Q_hat(s,a) + b(s)
    Used for Stage2/Stage3.
    """
    def __init__(self, q_hat: QNetwork, b_net: VNetwork):
        super().__init__()
        self.q_hat = q_hat
        self.b_net = b_net

    def forward(self, obs: ObsType, action: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            q = _qhat_value(self.q_hat, obs, action).view(-1)
        b = self.b_net(obs).view(-1)
        return q + b

    def q_value(self, obs: ObsType, action: torch.Tensor) -> torch.Tensor:
        return self.forward(obs, action)


def _gumbel_rescale_loss(diff: torch.Tensor, alpha: float, max_clip: Optional[float] = None) -> torch.Tensor:
    z = diff / float(alpha)
    if max_clip is not None:
        max_clip_t = torch.as_tensor(max_clip, dtype=z.dtype, device=z.device)
        z = torch.minimum(z, max_clip_t)

    max_z = torch.max(z)
    max_z = torch.where(max_z < -1.0, torch.full_like(max_z, -1.0), max_z)
    max_z = max_z.detach()
    loss = torch.exp(z - max_z) - z * torch.exp(-max_z) - torch.exp(-max_z)
    return loss


def _expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    w_pos = float(expectile)
    w_neg = 1.0 - w_pos
    weight = torch.where(diff > 0, w_pos, w_neg)
    return weight * diff.pow(2)


def _grad_gumbel(x: torch.Tensor, alpha: float, clip_max: float = 7.0) -> torch.Tensor:
    if clip_max is not None:
        clip_max_t = torch.as_tensor(clip_max, dtype=x.dtype, device=x.device)
        x = torch.minimum(x, clip_max_t)

    x_max = torch.max(x)
    x_max = torch.where(x_max < -1.0, torch.full_like(x_max, -1.0), x_max)

    x1 = x - x_max
    num = torch.exp(x1) - torch.exp(-x_max)
    den = torch.mean(torch.exp(x1) - x * torch.exp(-x_max), dim=0, keepdim=True)
    grad = num / (den + 1e-8)
    return grad


def _gumbel_log_loss(diff: torch.Tensor, alpha: float, clip_max: float = 7.0) -> torch.Tensor:
    x = diff / float(alpha)
    grad = _grad_gumbel(x, alpha=alpha, clip_max=clip_max)
    grad_detached = grad.detach()
    loss = grad_detached * x
    return loss


def _augment_obs_and_actions_for_v(
    obs: ObsType,
    action: torch.Tensor,
    loss_cfg: Dict[str, Any],
    device: torch.device,
) -> Tuple[ObsType, torch.Tensor]:
    sample_random_times = int(loss_cfg.get("sample_random_times", 0))
    use_noise = bool(loss_cfg.get("noise", False))
    noise_std = float(loss_cfg.get("noise_std", 0.0))
    noise_clip = float(loss_cfg.get("noise_clip", 0.5))
    action_low = float(loss_cfg.get("action_low", -1.0))
    action_high = float(loss_cfg.get("action_high", 1.0))

    acts = action.to(device=device, dtype=torch.float32)
    if acts.dim() == 1:
        acts = acts.view(1, -1)
    B, A = acts.shape

    if isinstance(obs, dict):
        obs_t = {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in obs.items()}
    else:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)

    obs_aug = obs_t
    acts_aug = acts

    if sample_random_times > 0:
        times = sample_random_times
        rand_acts = torch.empty(times * B, A, device=device).uniform_(action_low, action_high)

        if isinstance(obs_t, dict):
            obs_rand = {}
            for k, v in obs_t.items():
                reps = (times,) + (1,) * (v.dim() - 1)
                obs_rand[k] = v.repeat(reps)
            obs_cat = {k: torch.cat([obs_t[k], obs_rand[k]], dim=0) for k in obs_t.keys()}
        else:
            reps = (times,) + (1,) * (obs_t.dim() - 1)
            obs_rand = obs_t.repeat(reps)
            obs_cat = torch.cat([obs_t, obs_rand], dim=0)

        obs_aug = obs_cat
        acts_aug = torch.cat([acts, rand_acts], dim=0)

    if use_noise and noise_std > 0.0:
        noise = torch.randn_like(acts_aug, device=device) * noise_std
        if noise_clip is not None and noise_clip > 0:
            noise = torch.clamp(noise, -noise_clip, noise_clip)
        acts_aug = acts_aug + noise
        acts_aug = torch.clamp(acts_aug, action_low, action_high)

    return obs_aug, acts_aug


def train_v_function_step(
    *,
    v_net: VNetwork,
    v_optim: torch.optim.Optimizer,
    q_full: nn.Module,
    obs: ObsType,
    action: torch.Tensor,
    cfg: dict,
) -> float:
    v_net.train()
    loss_cfg = cfg.get("loss", {}) if isinstance(cfg, dict) else {}
    loss_type = str(loss_cfg.get("type", "gumbel")).lower()
    beta = float(loss_cfg.get("beta", 1.0))
    expectile = float(loss_cfg.get("expectile", 0.8))
    max_clip = loss_cfg.get("max_clip", None)
    log_loss = bool(loss_cfg.get("log_loss", False))
    clip_max = float(loss_cfg.get("clip_max", 7.0))

    device = next(v_net.parameters()).device

    obs_aug, acts_aug = _augment_obs_and_actions_for_v(
        obs=obs,
        action=action,
        loss_cfg=loss_cfg,
        device=device,
    )

    with torch.no_grad():
        q_hat = q_full(obs_aug, acts_aug).view(-1).detach()

    v_values = v_net(obs_aug).view(-1)
    diff = q_hat - v_values

    if loss_type == "expectile":
        loss_vec = _expectile_loss(diff, expectile=expectile)
    else:
        if log_loss:
            loss_vec = _gumbel_log_loss(diff, alpha=beta, clip_max=clip_max)
        else:
            loss_vec = _gumbel_rescale_loss(diff, alpha=beta, max_clip=max_clip)

    loss_v = loss_vec.mean()

    v_optim.zero_grad(set_to_none=True)
    loss_v.backward()
    v_optim.step()

    return float(loss_v.item())


def train_reward_function_step(
    *,
    r_net: RewardNetwork,
    r_optim: torch.optim.Optimizer,
    q_targ: nn.Module,    # <-- QFullWrapper
    v_targ: VNetwork,
    obs: ObsType,
    action: torch.Tensor,
    next_obs: ObsType,
    done_t: torch.Tensor,
    gamma: float,
    r_loss_cfg: dict,
    r_rms: RunningMeanStd,
) -> Tuple[float, float]:
    r_loss_cfg = r_loss_cfg or {}
    r_clip = float(r_loss_cfg.get("target_clip", 1e6))
    huber_delta = float(r_loss_cfg.get("huber_delta", 1.0))
    clip_grad = float(r_loss_cfg.get("clip_grad_norm", 1.0))
    use_huber = bool(r_loss_cfg.get("use_huber", True))
    standardize = bool(r_loss_cfg.get("standardize_target", True))

    r_net.train()
    device = next(r_net.parameters()).device
    done_f = done_t.to(device=device, dtype=torch.float32).view(-1)

    with torch.no_grad():
        q_sa = q_targ(obs, action).view(-1)
        v_sp = v_targ(next_obs).view(-1)
        target_r = q_sa - float(gamma) * (1.0 - done_f) * v_sp
        if math.isfinite(r_clip):
            target_r = torch.clamp(target_r, -r_clip, r_clip)

    target_r_raw = target_r.detach()

    if standardize and (r_rms is not None):
        r_rms.update(target_r)
        mean, std = r_rms.mean, r_rms.std
        target_r = (target_r - mean) / (std + 1e-6)

    r_pred = r_net(obs, action).view(-1)
    if standardize and (r_rms is not None):
        mean, std = r_rms.mean, r_rms.std
        r_pred_std = (r_pred - mean) / (std + 1e-6)
        pred_for_loss = r_pred_std
    else:
        pred_for_loss = r_pred

    if use_huber:
        train_loss = F.smooth_l1_loss(pred_for_loss, target_r.view(-1), beta=huber_delta)
    else:
        train_loss = F.mse_loss(pred_for_loss, target_r.view(-1))

    unnorm_mse = F.mse_loss(r_pred, target_r_raw.view(-1))

    r_optim.zero_grad(set_to_none=True)
    train_loss.backward()
    if clip_grad and clip_grad > 0:
        torch.nn.utils.clip_grad_norm_(r_net.parameters(), clip_grad)
    r_optim.step()

    return float(train_loss.item()), float(unnorm_mse.item())


def main():
    parser = argparse.ArgumentParser("LFIRL pipeline (Qhat, initial V, b-offset, calibrated V, reward)")
    parser.add_argument(
        "--cfg",
        type=str,
        default=os.path.join("environments", "AdroitHandPen-v1.yml"),
        help="Path to environment YAML",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.cfg)
    env_id = cfg["env_id"]
    p = cfg["policy"]
    paths = cfg["paths"]

    q_cfg = cfg.get("q_net", {})
    b_cfg = cfg.get("b_net", {})
    v_cfg = cfg.get("v_net", {})
    r_cfg = cfg.get("r_net", {})
    irl_cfg = cfg.get("irl", {})
    online_cfg = cfg.get("online", {})

    q_loss_cfg = q_cfg.get("loss", {}) if isinstance(q_cfg, dict) else {}
    b_loss_cfg = b_cfg.get("loss", {}) if isinstance(b_cfg, dict) else {}
    r_loss_cfg = r_cfg.get("loss", {}) if isinstance(r_cfg, dict) else {}

    gamma = float(irl_cfg.get("gamma", 0.99))
    ema_tau = float(irl_cfg.get("ema_tau", 0.005))
    batch_size = int(irl_cfg.get("batch_size", p.get("batch_size", 256)))

    # stage passes (new names + backward compatible)
    qhat_passes = int(irl_cfg.get("qhat_passes", irl_cfg.get("q_passes", 1)))
    b_passes = int(irl_cfg.get("b_passes", irl_cfg.get("stage15_passes", 1)))
    v_passes = int(irl_cfg.get("v_passes", 1))
    initial_v_passes = int(irl_cfg.get("initial_v_passes", irl_cfg.get("v_init_passes", v_passes)))
    calibrated_v_passes = int(
        irl_cfg.get("calibrated_v_passes", irl_cfg.get("v_recalib_passes", v_passes))
    )
    r_passes = int(irl_cfg.get("r_passes", 1))
    enable_b_training_for_flow_matching = bool(
        irl_cfg.get("enable_b_training_for_flow_matching", True)
    )
    use_online_vr = bool(online_cfg.get("enabled", False))

    dp_norm_mode = str(irl_cfg.get("dp_norm_mode", "gaussian"))
    dp_norm_max = int(irl_cfg.get("dp_norm_max_samples", 200000))

    ensure_dirs(paths["pretrained_dir"], paths["expert_dir"], paths["results_dir"])

    env = make_env(env_id, seed=p["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_str = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    print(f"[Setup] Env: {env_id} | Device: {dev_str}")

    logger = Logger(env_id=env_id, results_root=paths["results_dir"])
    start_time = time.time()

    offline = OfflineDataset(env_id, env=env, num_expert=p["num_expert"], seed=p.get("seed", None))
    action_dim = offline.action_dim

    # policy obs flatten keys (for dict-obs env)
    policy_obs_keys: Optional[List[str]] = None
    if getattr(offline, "states_dict", None) is not None:
        policy_obs_keys = sorted(list(offline.states_dict.keys()))
        print(f"[Offline] dict-obs keys for policy flatten: {policy_obs_keys}")

    if hasattr(env.action_space, "low"):
        action_low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
        action_high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    else:
        # fallback for discrete (not your case)
        action_low = -np.ones((action_dim,), dtype=np.float32)
        action_high = np.ones((action_dim,), dtype=np.float32)

    policy_backend = resolve_policy_backend(p)
    skip_b_training_for_flow_matching = (
        policy_backend == "flow_matching"
        and not enable_b_training_for_flow_matching
    )
    effective_b_passes = 0 if skip_b_training_for_flow_matching else b_passes
    b_is_calibrated = effective_b_passes > 0

    if skip_b_training_for_flow_matching:
        print(
            "[Ablation] Flow Matching with b(s) alignment disabled: "
            "skip Stage2-b and fall back to Qhat-only value/reward targets."
        )
    else:
        print(
            f"[Config] b(s) alignment enabled for backend={policy_backend} "
            f"(passes={effective_b_passes})."
        )

    policy_ckpt, policy_ckpt_candidates = get_policy_checkpoint_candidates(
        env_id=env_id,
        pretrained_dir=paths["pretrained_dir"],
        policy_cfg=p,
    )
    existing_policy_ckpt = next((path for path in policy_ckpt_candidates if os.path.exists(path)), None)
    if existing_policy_ckpt is not None:
        if getattr(offline, "states_dict", None) is not None:
            obs_arr = np.concatenate([offline.states_dict[k] for k in policy_obs_keys], axis=-1)
            obs_dim = int(obs_arr.shape[-1])
        else:
            obs_dim = int(offline.states.shape[-1])

        print(
            f"[Policy] backend={policy_backend} | obs_dim={obs_dim} (dict-obs concatenated)"
            if policy_obs_keys
            else f"[Policy] backend={policy_backend} | obs_dim={obs_dim}"
        )

        model = build_diffusion_policy_model(
            obs_dim=obs_dim,
            action_dim=action_dim,
            backend=policy_backend,
            device=device,
            horizon=int(p["pred_horizon"]),
            n_obs_steps=int(p["obs_horizon"]),
            n_action_steps=int(p["pred_horizon"]),
            num_train_timesteps=int(p["timesteps"]),
            num_inference_steps=int(p["ddim_steps"]),
            obs_as_local_cond=False,
            obs_as_global_cond=True,
            pred_action_steps_only=True,
        )

        missing, unexpected = load_dp_with_normalizer_preload(
            dp_model=model,
            ckpt_path=existing_policy_ckpt,
            offline=offline,
            policy_obs_keys=policy_obs_keys,
            device=device,
            mode=dp_norm_mode,
            max_samples=dp_norm_max,
        )
        print(f"[Policy] Loaded pretrained {policy_backend} policy from {existing_policy_ckpt}")
        if missing:
            print(f"[Policy] Missing keys: {len(missing)} (ok if buffers): {missing[:10]} ...")
        if unexpected:
            # after preload, unexpected should NOT include normalizer.params_dict.*
            print(f"[Policy] Unexpected keys: {len(unexpected)} (ok): {unexpected[:10]} ...")
        model.eval()
    else:
        print(f"[Policy] No pretrained {policy_backend} policy found. Start pretraining...")
        model = train_diffusion_policy(env, cfg, device=device)
        torch.save(model.state_dict(), policy_ckpt)
        print(f"[Policy] Saved pretrained {policy_backend} policy to {policy_ckpt}")
        model.eval()

    policy_eval = evaluate_policy_receding(
        model=model,
        env=env,
        episodes=p["eval_episodes"],
        obs_horizon=p["obs_horizon"],
        pred_horizon=p["pred_horizon"],
        action_horizon=p["action_horizon"],
        device=device,
        base_seed=p["seed"],
    )
    print(f"[Eval] Diffusion Policy metric={policy_eval:.3f}")

    q_hat = QNetwork.build_q_from_cfg_with_env(
        cfg=q_cfg,
        env=env,
        obs_horizon=p["obs_horizon"],
        device=device,
        warmup_forward=True,
    )
    print("[Qhat] Built Qhat network.")

    b_net = VNetwork.build_v_from_cfg_with_env(
        cfg=b_cfg,
        env=env,
        obs_horizon=p["obs_horizon"],
        device=device,
        warmup_forward=True,
    )
    print("[b(s)] Built b network.")

    v_net = VNetwork.build_v_from_cfg_with_env(
        cfg=v_cfg,
        env=env,
        obs_horizon=p["obs_horizon"],
        device=device,
        warmup_forward=True,
    )
    print("[V] Built V network.")

    r_net = RewardNetwork.build_reward_from_cfg_with_env(
        cfg=r_cfg,
        env=env,
        obs_horizon=p["obs_horizon"],
        device=device,
        warmup_forward=True,
    )
    print("[R] Built reward network.")

    if getattr(offline, "states_dict", None) is not None:
        mean_d, std_d = compute_obs_stats_dict(offline.states_dict, indices=offline.train_indices)
        mean_d = {k: v.to(device) for k, v in mean_d.items()}
        std_d = {k: v.to(device) for k, v in std_d.items()}
        q_hat.set_obs_normalizer(mean_dict=mean_d, std_dict=std_d)
        b_net.set_obs_normalizer(mean_dict=mean_d, std_dict=std_d)
        v_net.set_obs_normalizer(mean_dict=mean_d, std_dict=std_d)
        r_net.set_obs_normalizer(mean_dict=mean_d, std_dict=std_d)
        print(f"[Norm] Dict-obs keys: {list(mean_d.keys())}")
    else:
        mean, std = compute_obs_stats_flat(offline.states, indices=offline.train_indices)
        mean, std = mean.to(device), std.to(device)
        q_hat.set_obs_normalizer(mean=mean, std=std)
        b_net.set_obs_normalizer(mean=mean, std=std)
        v_net.set_obs_normalizer(mean=mean, std=std)
        r_net.set_obs_normalizer(mean=mean, std=std)
        print(f"[Norm] Flat-obs: state_dim={offline.states.shape[1]}")

    q_lr = float(q_cfg.get("lr", 3e-4))
    q_wd = float(q_cfg.get("weight_decay", 0.0))
    b_lr = float(b_cfg.get("lr", 3e-4))
    b_wd = float(b_cfg.get("weight_decay", 0.0))
    v_lr = float(v_cfg.get("lr", 3e-4))
    v_wd = float(v_cfg.get("weight_decay", 0.0))
    r_lr = float(r_cfg.get("lr", 1e-4))
    r_wd = float(r_cfg.get("weight_decay", 0.0))

    q_optim = torch.optim.AdamW(q_hat.parameters(), lr=q_lr, weight_decay=q_wd)
    b_optim = torch.optim.AdamW(b_net.parameters(), lr=b_lr, weight_decay=b_wd)
    v_optim = torch.optim.AdamW(v_net.parameters(), lr=v_lr, weight_decay=v_wd)
    r_optim = torch.optim.AdamW(r_net.parameters(), lr=r_lr, weight_decay=r_wd)
    print("[Opt] Optimizers ready (Qhat/b/V/R).")

    qhat_targ = QNetwork.build_q_from_cfg_with_env(q_cfg, env, p["obs_horizon"], device=device, warmup_forward=True)
    b_targ = VNetwork.build_v_from_cfg_with_env(b_cfg, env, p["obs_horizon"], device=device, warmup_forward=True)
    v_targ = VNetwork.build_v_from_cfg_with_env(v_cfg, env, p["obs_horizon"], device=device, warmup_forward=True)

    qhat_targ.load_state_dict(q_hat.state_dict(), strict=True)
    b_targ.load_state_dict(b_net.state_dict(), strict=True)
    v_targ.load_state_dict(v_net.state_dict(), strict=True)

    r_rms = RunningMeanStd()

    To = int(p["obs_horizon"])
    states_src = offline.states_dict if getattr(offline, "states_dict", None) is not None else offline.states
    actions = offline.actions
    terminals = getattr(offline, "terminals", None)
    timeouts = getattr(offline, "timeouts", None)
    dones = getattr(offline, "dones", None)

    traj_slices = get_traj_slices_prefer_offline(offline)
    num_traj = len(traj_slices)

    offline_source = dict(
        name="offline_expert",
        states_src=states_src,
        actions=actions,
        terminals=terminals,
        timeouts=timeouts,
        dones=dones,
        traj_slices=traj_slices,
    )

    offline_steps_per_pass = _count_transitions_from_slices(traj_slices)
    print(f"[Offline] Num trajectories: {num_traj} | Total env steps (per pass): {offline_steps_per_pass}")
    if offline_steps_per_pass <= 0:
        raise RuntimeError("No transitions in offline dataset (offline_steps_per_pass == 0).")

    if use_online_vr:
        print("[LFIRL] online.enabled is ignored: the PDF pipeline trains V/b/R on the expert buffer only.")

    v_source = offline_source
    r_source = offline_source

    v_steps_per_pass = _count_transitions_from_slices(v_source["traj_slices"])
    r_steps_per_pass = _count_transitions_from_slices(r_source["traj_slices"])

    if (qhat_passes + initial_v_passes + effective_b_passes + calibrated_v_passes + r_passes) <= 0:
        raise RuntimeError("Total passes must be > 0.")
    if (initial_v_passes > 0 or calibrated_v_passes > 0) and v_steps_per_pass <= 0:
        raise RuntimeError("No transitions available for Stage2-V.")
    if r_passes > 0 and r_steps_per_pass <= 0:
        raise RuntimeError("No transitions available for Stage3-R.")

    total_batches = (
        qhat_passes * _count_batches(offline_steps_per_pass, batch_size)
        + initial_v_passes * _count_batches(v_steps_per_pass, batch_size)
        + effective_b_passes * _count_batches(offline_steps_per_pass, batch_size)
        + calibrated_v_passes * _count_batches(v_steps_per_pass, batch_size)
        + r_passes * _count_batches(r_steps_per_pass, batch_size)
    )
    if total_batches <= 0:
        raise RuntimeError("Total batches must be > 0.")

    num_logs = min(1000, total_batches)
    num_logs = max(num_logs, 1)
    checkpoints = [int((total_batches * k + num_logs - 1) // num_logs) for k in range(1, num_logs + 1)]
    cp_idx = 0
    flush_id = 0

    pbar = tqdm(total=num_logs, desc=f"Train@{env_id}", unit="log", dynamic_ncols=True)

    env_steps_so_far = 0
    last_q, last_b, last_v, last_r = float("nan"), float("nan"), float("nan"), float("nan")

    def iter_transitions(source: Dict[str, Any]):
        source_states = source["states_src"]
        source_actions = source["actions"]
        source_traj_slices = source["traj_slices"]
        source_terminals = source.get("terminals", None)
        source_timeouts = source.get("timeouts", None)
        source_dones = source.get("dones", None)

        for (start, end_exclusive) in source_traj_slices:
            for t in range(start, end_exclusive - 1):
                obs_t = windowize_obs(source_states, t, start_idx=start, To=To, device=device)
                obs_tp1 = windowize_obs(source_states, t + 1, start_idx=start, To=To, device=device)
                a_t = make_action_tensor(source_actions[t], action_dim=action_dim, device=device)

                if (source_terminals is not None) or (source_timeouts is not None):
                    term = bool(source_terminals[t]) if source_terminals is not None else False
                    tout = bool(source_timeouts[t]) if source_timeouts is not None else False
                    done_t = term or tout
                elif source_dones is not None:
                    done_t = bool(source_dones[t])
                else:
                    done_t = (t == end_exclusive - 2)

                yield obs_t, obs_tp1, a_t, done_t

    def flush_stage_batch(stage: str, obs_batch, obs_next_batch, act_batch, done_batch):
        nonlocal env_steps_so_far, last_q, last_b, last_v, last_r, flush_id, cp_idx

        if len(act_batch) == 0:
            return

        obs_b = _collate_obs_list(obs_batch)
        obs_next_b = _collate_obs_list(obs_next_batch)

        first_a = act_batch[0]
        if first_a.dim() == 2 and first_a.size(0) == 1:
            act_b = torch.cat(act_batch, dim=0)
        elif first_a.dim() == 1:
            act_b = torch.stack(act_batch, dim=0)
        else:
            act_b = torch.stack([a.view(-1) for a in act_batch], dim=0)
        act_b = act_b.to(device=device)

        done_b = torch.tensor(done_batch, dtype=torch.float32, device=device).view(-1)

        if stage == "QHAT":
            last_q = train_qhat_LFIRL_step(
                q_hat=q_hat,
                q_optim=q_optim,
                dp_model=model,
                obs=obs_b,
                action=act_b,
                device=device,
                policy_obs_keys=policy_obs_keys,
                q_loss_cfg=q_loss_cfg,
                action_low=action_low,
                action_high=action_high,
            )
            soft_update(qhat_targ, q_hat, tau=ema_tau)

        elif stage == "B":
            # freeze Qhat and the initial V for offset calibration.
            q_hat.eval()
            v_net.eval()
            for p_ in q_hat.parameters():
                p_.requires_grad_(False)
            for p_ in v_net.parameters():
                p_.requires_grad_(False)

            last_b = train_b_offset_step(
                q_hat=q_hat,
                v_net=v_net,
                b_net=b_net,
                b_optim=b_optim,
                obs=obs_b,
                action=act_b,
                next_obs=obs_next_b,
                done_t=done_b,
                gamma=gamma,
                device=device,
                action_dim=action_dim,
                action_low=action_low,
                action_high=action_high,
                cfg=b_cfg,
            )
            soft_update(b_targ, b_net, tau=ema_tau)

        elif stage == "V_INIT":
            # Eq. (10): fit initial V from frozen uncalibrated Qhat.
            q_hat.eval()
            for p_ in q_hat.parameters():
                p_.requires_grad_(False)
            for p_ in v_net.parameters():
                p_.requires_grad_(True)

            q_view = QHatOnlyWrapper(q_hat)
            last_v = train_v_function_step(
                v_net=v_net,
                v_optim=v_optim,
                q_full=q_view,
                obs=obs_b,
                action=act_b,
                cfg=v_cfg,
            )
            soft_update(v_targ, v_net, tau=ema_tau)

        elif stage == "V_CALIBRATED":
            # Eq. (12): refit V after adding the learned state-dependent offset.
            q_hat.eval()
            b_net.eval()
            for p_ in q_hat.parameters():
                p_.requires_grad_(False)
            for p_ in b_net.parameters():
                p_.requires_grad_(False)
            for p_ in v_net.parameters():
                p_.requires_grad_(True)

            q_full = QFullWrapper(q_hat, b_net) if b_is_calibrated else QHatOnlyWrapper(q_hat)
            last_v = train_v_function_step(
                v_net=v_net,
                v_optim=v_optim,
                q_full=q_full,
                obs=obs_b,
                action=act_b,
                cfg=v_cfg,
            )
            soft_update(v_targ, v_net, tau=ema_tau)

        elif stage == "R":
            # use EMA targets: qhat_targ + b_targ, and v_targ
            q_full_targ = QFullWrapper(qhat_targ, b_targ) if b_is_calibrated else QHatOnlyWrapper(qhat_targ)
            train_rloss, _ = train_reward_function_step(
                r_net=r_net,
                r_optim=r_optim,
                q_targ=q_full_targ,
                v_targ=v_targ,
                obs=obs_b,
                action=act_b,
                next_obs=obs_next_b,
                done_t=done_b,
                gamma=gamma,
                r_loss_cfg=r_loss_cfg,
                r_rms=r_rms,
            )
            last_r = train_rloss

        else:
            raise ValueError(f"Unknown stage: {stage}")

        env_steps_so_far += len(act_batch)
        flush_id += 1

        while cp_idx < num_logs and flush_id >= checkpoints[cp_idx]:
            logger.log(env_steps_so_far, last_q, last_v, last_r)
            pbar.set_postfix(
                env_steps=env_steps_so_far,
                qhat=(f"{last_q:.4f}" if last_q == last_q else "nan"),
                b=(f"{last_b:.4f}" if last_b == last_b else "nan"),
                v=(f"{last_v:.4f}" if last_v == last_v else "nan"),
                r=(f"{last_r:.4f}" if last_r == last_r else "nan"),
            )
            pbar.update(1)
            cp_idx += 1

        obs_batch.clear()
        obs_next_batch.clear()
        act_batch.clear()
        done_batch.clear()

    if qhat_passes > 0:
        pbar.set_description(f"Train-Qhat@{env_id}")
        print(f"[Stage1-Qhat] Passes: {qhat_passes} | Source: {offline_source['name']}")
        for ep in range(qhat_passes):
            obs_batch, obs_next_batch, act_batch, done_batch = [], [], [], []
            for obs_t, obs_tp1, a_t, done_t in iter_transitions(offline_source):
                obs_batch.append(obs_t)
                obs_next_batch.append(obs_tp1)
                act_batch.append(a_t)
                done_batch.append(done_t)
                if len(act_batch) >= batch_size:
                    flush_stage_batch("QHAT", obs_batch, obs_next_batch, act_batch, done_batch)
            flush_stage_batch("QHAT", obs_batch, obs_next_batch, act_batch, done_batch)

    if initial_v_passes > 0:
        pbar.set_description(f"Train-Vinit@{env_id}")
        print(f"[Stage1-Vinit] Passes: {initial_v_passes} | Source: {v_source['name']}")
        for ep in range(initial_v_passes):
            obs_batch, obs_next_batch, act_batch, done_batch = [], [], [], []
            for obs_t, obs_tp1, a_t, done_t in iter_transitions(v_source):
                obs_batch.append(obs_t)
                obs_next_batch.append(obs_tp1)
                act_batch.append(a_t)
                done_batch.append(done_t)
                if len(act_batch) >= batch_size:
                    flush_stage_batch("V_INIT", obs_batch, obs_next_batch, act_batch, done_batch)
            flush_stage_batch("V_INIT", obs_batch, obs_next_batch, act_batch, done_batch)

    if effective_b_passes > 0:
        pbar.set_description(f"Train-b@{env_id}")
        print(f"[Stage2-b] Passes: {effective_b_passes} | Source: {offline_source['name']}")
        for ep in range(effective_b_passes):
            obs_batch, obs_next_batch, act_batch, done_batch = [], [], [], []
            for obs_t, obs_tp1, a_t, done_t in iter_transitions(offline_source):
                obs_batch.append(obs_t)
                obs_next_batch.append(obs_tp1)
                act_batch.append(a_t)
                done_batch.append(done_t)
                if len(act_batch) >= batch_size:
                    flush_stage_batch("B", obs_batch, obs_next_batch, act_batch, done_batch)
            flush_stage_batch("B", obs_batch, obs_next_batch, act_batch, done_batch)
    elif skip_b_training_for_flow_matching:
        print("[Stage2-b] Skipped for Flow Matching ablation.")

    if calibrated_v_passes > 0:
        pbar.set_description(f"Train-Vcal@{env_id}")
        print(f"[Stage2-Vcal] Passes: {calibrated_v_passes} | Source: {v_source['name']}")
        for ep in range(calibrated_v_passes):
            obs_batch, obs_next_batch, act_batch, done_batch = [], [], [], []
            for obs_t, obs_tp1, a_t, done_t in iter_transitions(v_source):
                obs_batch.append(obs_t)
                obs_next_batch.append(obs_tp1)
                act_batch.append(a_t)
                done_batch.append(done_t)
                if len(act_batch) >= batch_size:
                    flush_stage_batch("V_CALIBRATED", obs_batch, obs_next_batch, act_batch, done_batch)
            flush_stage_batch("V_CALIBRATED", obs_batch, obs_next_batch, act_batch, done_batch)

    # Use the finalized frozen stage outputs for Bellman reward regression.
    if r_passes > 0:
        qhat_targ.load_state_dict(q_hat.state_dict(), strict=True)
        b_targ.load_state_dict(b_net.state_dict(), strict=True)
        v_targ.load_state_dict(v_net.state_dict(), strict=True)

    if r_passes > 0:
        pbar.set_description(f"Train-R@{env_id}")
        print(f"[Stage3-R] Passes: {r_passes} | Source: {r_source['name']}")
        for ep in range(r_passes):
            obs_batch, obs_next_batch, act_batch, done_batch = [], [], [], []
            for obs_t, obs_tp1, a_t, done_t in iter_transitions(r_source):
                obs_batch.append(obs_t)
                obs_next_batch.append(obs_tp1)
                act_batch.append(a_t)
                done_batch.append(done_t)
                if len(act_batch) >= batch_size:
                    flush_stage_batch("R", obs_batch, obs_next_batch, act_batch, done_batch)
            flush_stage_batch("R", obs_batch, obs_next_batch, act_batch, done_batch)

    while cp_idx < num_logs:
        logger.log(env_steps_so_far, last_q, last_v, last_r)
        pbar.update(1)
        cp_idx += 1
    pbar.close()

    elapsed = time.time() - start_time
    logger.finalize(elapsed)

    # Save models safely (don't assume Logger accepts b_net)
    try:
        logger.save_models(q_net=q_hat, v_net=v_net, r_net=r_net)
    except Exception:
        pass

    # Always save b_net separately
    b_path = os.path.join(paths["results_dir"], f"{sanitize_env_id(env_id)}_b_net.pt")
    torch.save(b_net.state_dict(), b_path)
    print(f"[Save] b_net -> {b_path}")

    eval_mean = evaluate_reward_via_sac(
        env_id=env_id,
        make_env_fn=make_env,
        reward_net=r_net,
        total_timesteps=250_000,
        eval_episodes=100,
        seed=p["seed"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        sb3_kwargs=dict(batch_size=256, learning_rate=3e-4),
    )
    print("[SAC-Eval] mean return on ORIGINAL env =", eval_mean)
    logger.log_reward(eval_mean)

    print(f"[Done] Total transitions per pass = {offline_steps_per_pass}")
    print(
        f"[Done] Passes: Qhat={qhat_passes}, Vinit={initial_v_passes}, "
        f"b={effective_b_passes}, Vcal={calibrated_v_passes}, R={r_passes}"
    )
    print(f"[Done] Total transition updates (all stages) = {env_steps_so_far}")
    print(f"[Done] Total wall-clock time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()




