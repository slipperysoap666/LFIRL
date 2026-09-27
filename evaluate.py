#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import yaml
import gym_pusht

# Project imports

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.r_network import RewardNetwork
from models.q_network import ObsEncoder

try:
    from utils.utils import make_env as project_make_env
except Exception:
    project_make_env = None


# Constants

FLAT_KEY_SEP = "__"
TIMESTAMP_DIR_RE = re.compile(r"^\d{8}_\d{6}$")

# For AdroitHandPen / Pen-like environments, generated trajectories are only accepted
# when their true environment cumulative reward is no higher than this threshold.
PEN_GENERATED_MAX_CUMULATIVE_REWARD = -500.0


# Path / env helpers

def _safe_id_for_filename(s: str) -> str:
    s = str(s)
    s = re.sub(r'[<>:"/\\|?*]+', "__", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s if s else "env"


def _infer_env_tag_from_reward_path(reward_path: Union[str, Path]) -> str:
    p = Path(reward_path)
    parts = p.parts

    if len(parts) >= 3 and TIMESTAMP_DIR_RE.match(parts[-2]):
        return parts[-3]

    parent = p.parent
    if TIMESTAMP_DIR_RE.match(parent.name) and parent.parent.name:
        return parent.parent.name

    for part in reversed(parts[:-1]):
        if re.search(r"-v\d+$", part):
            return part

    if parent.name:
        return parent.name

    raise RuntimeError(f"Could not infer environment tag from reward path: {reward_path}")


def _recover_env_id_from_env_tag(env_tag: str) -> Optional[str]:
    """
    Recover a real gym env id from a run-folder tag.

    Supported:
      1) tags already using safe replacement:
         "gym_pusht__PushT-v0" -> "gym_pusht/PushT-v0"

      2) tags flattened by replacing '/' with '_' for gym_* style packages:
         "gym_pusht_PushT-v0" -> "gym_pusht/PushT-v0"
         "gym_xarm_XarmLift-v0" -> "gym_xarm/XarmLift-v0"

    Important:
      - We only apply the "_" -> "/" recovery for tags that start with "gym_"
        and match the pattern "<package>_<task-vN>".
      - This avoids breaking ids like "PointMaze_UMaze-v3".
    """
    env_tag = str(env_tag)

    if "__" in env_tag:
        return env_tag.replace("__", "/")

    m = re.match(r"^(gym_[^_]+)_(.+-v\d+)$", env_tag)
    if m is not None:
        pkg = m.group(1)
        task = m.group(2)
        return f"{pkg}/{task}"

    return None


def _candidate_env_ids_from_tag(env_tag: str, env_id_override: Optional[str] = None) -> List[str]:
    candidates: List[str] = []

    if env_id_override:
        candidates.append(env_id_override)

    recovered = _recover_env_id_from_env_tag(env_tag)
    if recovered is not None:
        candidates.append(recovered)

    candidates.append(env_tag)

    seen = set()
    out: List[str] = []
    for x in candidates:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _make_env_from_candidates(env_candidates: Sequence[str]) -> Tuple[gym.Env, str]:
    last_error: Optional[Exception] = None

    for env_id in env_candidates:
        try:
            if project_make_env is not None:
                env = project_make_env(env_id)
            else:
                env = gym.make(env_id)
            return env, env_id
        except Exception as e:
            last_error = e

    raise RuntimeError(
        "Failed to create environment from all candidates:\n"
        f"  candidates={list(env_candidates)}\n"
        f"  last_error={repr(last_error)}"
    )


def _get_env_id(env: gym.Env) -> str:
    """Best-effort extraction of a Gymnasium environment id."""
    spec = getattr(env, "spec", None)
    env_id = getattr(spec, "id", None)
    if env_id:
        return str(env_id)

    unwrapped = getattr(env, "unwrapped", None)
    spec = getattr(unwrapped, "spec", None)
    env_id = getattr(spec, "id", None)
    if env_id:
        return str(env_id)

    return env.__class__.__name__


def _is_pen_env(env: gym.Env) -> bool:
    """Return True for AdroitHandPen / Pen-like environment ids."""
    env_id = _get_env_id(env).lower()
    return "pen" in env_id


def _find_expert_npz(env_tag: str, env_id_override: Optional[str] = None) -> Path:
    """
    Find expert_data/{tag}_trajectories.npz.

    Important for PushT-like cases:
      reward folder tag may be:  gym_pusht_PushT-v0
      real env id is:            gym_pusht/PushT-v0
      safe file name should be:  gym_pusht__PushT-v0_trajectories.npz
    """
    expert_dir = PROJECT_ROOT / "expert_data"

    candidate_tags: List[str] = []

    if env_id_override:
        candidate_tags.append(_safe_id_for_filename(env_id_override))

    candidate_tags.append(env_tag)

    safe_tag = _safe_id_for_filename(env_tag)
    if safe_tag not in candidate_tags:
        candidate_tags.append(safe_tag)

    recovered_env_id = _recover_env_id_from_env_tag(env_tag)
    if recovered_env_id is not None:
        recovered_safe_tag = _safe_id_for_filename(recovered_env_id)
        if recovered_safe_tag not in candidate_tags:
            candidate_tags.append(recovered_safe_tag)

    for tag in candidate_tags:
        p = expert_dir / f"{tag}_trajectories.npz"
        if p.exists():
            return p

    raise FileNotFoundError(
        "Could not find the expert trajectory file. Tried:\n"
        + "\n".join(str(expert_dir / f"{tag}_trajectories.npz") for tag in candidate_tags)
    )


# Observation helpers

def _flatten_nested_dict_leaves(
    d: Mapping[str, Any],
    prefix: str = "",
    sep: str = FLAT_KEY_SEP,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k, v in d.items():
        kk = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_nested_dict_leaves(v, prefix=kk, sep=sep))
        else:
            out[kk] = np.asarray(v)
    return out


def _flatten_obs_to_vec(
    obs_like: Any,
    key_order: Optional[Tuple[str, ...]] = None,
    sep: str = FLAT_KEY_SEP,
) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]], Optional[Tuple[str, ...]]]:
    if isinstance(obs_like, dict):
        leaf = _flatten_nested_dict_leaves(obs_like, sep=sep)

        if key_order is None:
            key_order = tuple(sorted(leaf.keys()))
        else:
            if set(leaf.keys()) != set(key_order):
                raise RuntimeError(
                    f"Observation dict leaf keys changed. "
                    f"expected={list(key_order)} got={sorted(leaf.keys())}"
                )

        leaf_1d: Dict[str, np.ndarray] = {}
        parts: List[np.ndarray] = []
        for k in key_order:
            v = np.asarray(leaf[k], dtype=np.float32).reshape(-1)
            leaf_1d[k] = v
            parts.append(v)

        vec = np.concatenate(parts, axis=0).astype(np.float32)
        return vec, leaf_1d, key_order

    arr = np.asarray(obs_like, dtype=np.float32).reshape(-1)
    return arr, None, None


def _prepare_single_obs_for_mode(
    obs_raw: Any,
    input_mode: str,
    obs_keys: Optional[Tuple[str, ...]] = None,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    if input_mode == "flat":
        vec, _leaf, _ = _flatten_obs_to_vec(obs_raw, key_order=obs_keys, sep=FLAT_KEY_SEP)
        return vec.astype(np.float32)

    if input_mode == "dict":
        if not isinstance(obs_raw, dict):
            raise RuntimeError("The reward model expects dict observations, but env returned a non-dict obs.")
        leaf = _flatten_nested_dict_leaves(obs_raw, sep=FLAT_KEY_SEP)
        keys = tuple(sorted(leaf.keys())) if obs_keys is None else obs_keys
        if obs_keys is not None and set(leaf.keys()) != set(obs_keys):
            raise RuntimeError(
                f"Dict obs keys mismatch. expected={list(obs_keys)} got={sorted(leaf.keys())}"
            )
        return {k: np.asarray(leaf[k], dtype=np.float32).reshape(-1) for k in keys}

    raise ValueError(f"Unknown input_mode: {input_mode}")


def _stack_obs_list(
    obs_list: List[Union[np.ndarray, Dict[str, np.ndarray]]],
    input_mode: str,
    obs_keys: Optional[Tuple[str, ...]] = None,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    if len(obs_list) == 0:
        raise RuntimeError("Cannot stack an empty obs list.")

    if input_mode == "flat":
        return np.stack([np.asarray(x, dtype=np.float32) for x in obs_list], axis=0).astype(np.float32)

    if input_mode == "dict":
        first = obs_list[0]
        assert isinstance(first, dict)

        keys = obs_keys if obs_keys is not None else tuple(first.keys())
        out: Dict[str, np.ndarray] = {}
        for k in keys:
            out[k] = np.stack(
                [np.asarray(x[k], dtype=np.float32) for x in obs_list],  # type: ignore[index]
                axis=0,
            ).astype(np.float32)
        return out

    raise ValueError(f"Unknown input_mode: {input_mode}")


def _slice_obs_seq(
    obs_seq: Union[np.ndarray, Dict[str, np.ndarray]],
    lo: int,
    hi: int,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    if isinstance(obs_seq, dict):
        return {k: v[lo:hi] for k, v in obs_seq.items()}
    return obs_seq[lo:hi]


def _obs_to_torch(
    obs: Union[np.ndarray, Dict[str, np.ndarray]],
    device: torch.device,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    if isinstance(obs, dict):
        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in obs.items()}
    return torch.as_tensor(obs, dtype=torch.float32, device=device)


def _window_array_repeat_first(arr: np.ndarray, horizon: int) -> np.ndarray:
    """
    Convert [T, ...] -> [T, H, ...] by using a left-padded sliding window.
    Padding repeats the first frame.
    """
    arr = np.asarray(arr, dtype=np.float32)
    if horizon <= 1:
        return arr

    if arr.shape[0] <= 0:
        shape = (0, horizon) + arr.shape[1:]
        return np.zeros(shape, dtype=np.float32)

    T = int(arr.shape[0])
    out_shape = (T, horizon) + tuple(arr.shape[1:])
    out = np.empty(out_shape, dtype=np.float32)

    first = arr[0]
    for t in range(T):
        start = max(0, t - horizon + 1)
        chunk = arr[start:t + 1]
        pad = horizon - chunk.shape[0]
        if pad > 0:
            out[t, :pad] = first
            out[t, pad:] = chunk
        else:
            out[t] = chunk
    return out


def _window_obs_seq(
    obs_seq: Union[np.ndarray, Dict[str, np.ndarray]],
    input_mode: str,
    obs_horizon: int,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    if obs_horizon <= 1:
        return obs_seq

    if input_mode == "flat":
        assert isinstance(obs_seq, np.ndarray)
        return _window_array_repeat_first(obs_seq, obs_horizon)

    if input_mode == "dict":
        assert isinstance(obs_seq, dict)
        return {k: _window_array_repeat_first(v, obs_horizon) for k, v in obs_seq.items()}

    raise ValueError(f"Unknown input_mode: {input_mode}")


# Expert buffer loading

def _compute_episode_ranges(dones: np.ndarray) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    start = 0
    n = int(dones.shape[0])

    for i in range(n):
        if bool(dones[i]):
            ranges.append((start, i))
            start = i + 1

    if start < n:
        ranges.append((start, n - 1))

    return ranges


def _load_expert_buffer(npz_path: Union[str, Path]) -> Dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)

    states = np.asarray(data["states"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    dones = np.asarray(data["dones"], dtype=np.bool_)

    obs_keys: Optional[Tuple[str, ...]] = None
    states_dict: Optional[Dict[str, np.ndarray]] = None

    if "obs_keys" in data.files:
        keys = tuple(data["obs_keys"].tolist())
        tmp_dict: Dict[str, np.ndarray] = {}
        ok = True
        for k in keys:
            name = f"states__{k}"
            if name not in data.files:
                ok = False
                break
            tmp_dict[k] = np.asarray(data[name], dtype=np.float32)

        if ok and len(tmp_dict) > 0:
            obs_keys = keys
            states_dict = tmp_dict

    episodes = _compute_episode_ranges(dones)

    return {
        "states": states,
        "actions": actions,
        "dones": dones,
        "obs_keys": obs_keys,
        "states_dict": states_dict,
        "episodes": episodes,
    }


def _sample_episode_ids(num_episodes: int, num_to_sample: int, rng: np.random.Generator) -> np.ndarray:
    if num_episodes <= 0:
        raise RuntimeError("No expert episodes available.")
    replace = num_episodes < num_to_sample
    return rng.choice(num_episodes, size=num_to_sample, replace=replace)


# Config helpers

def _find_reward_cfg_file(reward_path: Union[str, Path], explicit_cfg_path: Optional[str] = None) -> Optional[Path]:
    if explicit_cfg_path is not None:
        p = Path(explicit_cfg_path).resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"Explicit cfg file not found: {p}")

    run_dir = Path(reward_path).resolve().parent
    candidates = [
        run_dir / ".hydra" / "config.yaml",
        run_dir / ".hydra" / "config.yml",
        run_dir / "config.yaml",
        run_dir / "config.yml",
        run_dir / "cfg.yaml",
        run_dir / "cfg.yml",
        run_dir / "hparams.yaml",
        run_dir / "hparams.yml",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_yaml_dict(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"YAML file is not a dict: {path}")
    return data


def _find_reward_cfg_node(d: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(d, dict):
        return None

    if "r_net" in d and isinstance(d["r_net"], dict):
        return d

    direct_keys = {"state_embed_dim", "include_action", "hidden", "layers", "obs_encoder"}
    if any(k in d for k in direct_keys):
        return d

    for v in d.values():
        found = _find_reward_cfg_node(v)
        if found is not None:
            return found

    return None


def _load_cfg_hints(cfg_path: Optional[Path]) -> Dict[str, Any]:
    """
    Load optional non-shape hints from YAML.
    Shapes are still inferred from the checkpoint.
    """
    hints: Dict[str, Any] = {
        "reward_act": "silu",
        "encoder_act": "silu",
        "reward_layernorm": None,
        "encoder_layernorm": None,
        "encoder_agg": None,
        "encoder_concat_proj_dim": None,
        "encoder_use_goal_delta": None,
    }

    if cfg_path is None:
        return hints

    raw_cfg = _load_yaml_dict(cfg_path)
    cfg_node = _find_reward_cfg_node(raw_cfg)
    if cfg_node is None:
        return hints

    rcfg = cfg_node["r_net"] if ("r_net" in cfg_node and isinstance(cfg_node["r_net"], dict)) else cfg_node
    enc_cfg = rcfg.get("obs_encoder", {}) if isinstance(rcfg.get("obs_encoder", {}), dict) else {}

    if "act" in rcfg:
        hints["reward_act"] = str(rcfg["act"])
    if "layernorm" in rcfg:
        hints["reward_layernorm"] = bool(rcfg["layernorm"])

    if "act" in enc_cfg:
        hints["encoder_act"] = str(enc_cfg["act"])
    if "layernorm" in enc_cfg:
        hints["encoder_layernorm"] = bool(enc_cfg["layernorm"])
    if "agg" in enc_cfg:
        hints["encoder_agg"] = str(enc_cfg["agg"]).lower()
    if "concat_proj_dim" in enc_cfg:
        hints["encoder_concat_proj_dim"] = int(enc_cfg["concat_proj_dim"])
    if "use_goal_delta" in enc_cfg:
        hints["encoder_use_goal_delta"] = bool(enc_cfg["use_goal_delta"])

    return hints


# State_dict parsing / reconstruction

def _load_torch_object(path: Union[str, Path], device: torch.device):
    path = str(path)
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _strip_common_prefix_from_state_dict(sd: Mapping[str, torch.Tensor]) -> "OrderedDict[str, torch.Tensor]":
    sd = OrderedDict(sd)

    candidate_prefixes = [
        "",
        "module.",
        "model.",
        "r_net.",
        "reward_network.",
        "net.",
    ]

    best_sd = sd
    best_score = -1

    for prefix in candidate_prefixes:
        if prefix == "":
            trial = sd
        else:
            if not any(k.startswith(prefix) for k in sd.keys()):
                continue
            trial = OrderedDict()
            for k, v in sd.items():
                if k.startswith(prefix):
                    trial[k[len(prefix):]] = v
                else:
                    trial[k] = v

        score = sum(
            1
            for k in trial.keys()
            if k.startswith("mlp.") or k.startswith("obs_encoder.")
        )
        if score > best_score:
            best_score = score
            best_sd = trial

    return best_sd


def _extract_state_dict_from_checkpoint_obj(obj: Any) -> Optional["OrderedDict[str, torch.Tensor]"]:
    if isinstance(obj, nn.Module):
        return None

    if isinstance(obj, Mapping):
        tensor_values = sum(isinstance(v, torch.Tensor) for v in obj.values())
        if len(obj) > 0 and tensor_values > 0:
            return _strip_common_prefix_from_state_dict(obj)

        for key in ("state_dict", "model_state_dict", "r_net_state_dict"):
            if key in obj and isinstance(obj[key], Mapping):
                return _strip_common_prefix_from_state_dict(obj[key])

    return None


def _parse_top_level_reward_head(sd: Mapping[str, torch.Tensor], action_dim_from_data: int) -> List[Dict[str, Any]]:
    """
    Infer RewardNetwork MLP structure from mlp.* weights.
    Returns candidate constructor configs, trying include_action=True first.
    """
    linear_items: List[Tuple[int, torch.Tensor]] = []
    layernorm_found = False

    for k, v in sd.items():
        m = re.match(r"^mlp\.(\d+)\.weight$", k)
        if m is None:
            continue
        idx = int(m.group(1))
        if v.ndim == 2:
            linear_items.append((idx, v))
        elif v.ndim == 1:
            layernorm_found = True

    linear_items.sort(key=lambda x: x[0])

    if len(linear_items) == 0:
        raise RuntimeError("Could not infer reward head: no mlp.*.weight found.")

    linear_weights = [w for _, w in linear_items]
    first_linear = linear_weights[0]

    mlp_input_dim = int(first_linear.shape[1])
    hidden_dim = int(first_linear.shape[0]) if len(linear_weights) > 1 else 256
    num_linear_layers = int(len(linear_weights))

    candidates: List[Dict[str, Any]] = []

    if mlp_input_dim > action_dim_from_data:
        candidates.append(
            dict(
                include_action=True,
                action_dim=action_dim_from_data,
                state_embed_dim=mlp_input_dim - action_dim_from_data,
                hidden=hidden_dim,
                layers=num_linear_layers,
                layernorm=layernorm_found,
            )
        )

    candidates.append(
        dict(
            include_action=False,
            action_dim=0,
            state_embed_dim=mlp_input_dim,
            hidden=hidden_dim,
            layers=num_linear_layers,
            layernorm=layernorm_found,
        )
    )

    return candidates


def _collect_smallest_linear_weight_dims(sd: Mapping[str, torch.Tensor], prefix_regex: str) -> Optional[Tuple[int, int]]:
    """
    Return (out_dim, in_dim) for the smallest-index 2D weight matched by prefix_regex.
    """
    best_idx: Optional[int] = None
    best_shape: Optional[Tuple[int, int]] = None

    for k, v in sd.items():
        m = re.match(prefix_regex, k)
        if m is None:
            continue
        idx = int(m.group(1))
        if v.ndim != 2:
            continue
        if best_idx is None or idx < best_idx:
            best_idx = idx
            best_shape = (int(v.shape[0]), int(v.shape[1]))

    return best_shape


def _count_linear_layers_under(sd: Mapping[str, torch.Tensor], prefix_regex: str) -> int:
    count = 0
    for k, v in sd.items():
        m = re.match(prefix_regex, k)
        if m is None:
            continue
        if v.ndim == 2:
            count += 1
    return count


def _has_layernorm_weights_under(sd: Mapping[str, torch.Tensor], prefix_regex: str) -> bool:
    for k, v in sd.items():
        m = re.match(prefix_regex, k)
        if m is None:
            continue
        if v.ndim == 1 and k.endswith(".weight"):
            return True
    return False


def _infer_obs_encoder_meta(sd: Mapping[str, torch.Tensor], expert_buf: Dict[str, Any]) -> Dict[str, Any]:
    """
    Infer:
      - input_mode: "dict" or "flat"
      - obs_horizon
      - per-key / flat encoder layer widths
      - use_goal_delta
      - agg mode (best-effort)
    """
    states = expert_buf["states"]
    states_dict = expert_buf.get("states_dict", None)

    per_key_first_dims: Dict[str, Tuple[int, int]] = {}
    per_key_linear_counts: Dict[str, int] = {}
    per_key_layernorm: Dict[str, bool] = {}

    key_set = set()
    for k in sd.keys():
        m = re.match(r"^obs_encoder\._per_key_mlps\.([^\.]+)\.net\.(\d+)\.weight$", k)
        if m is not None:
            key_set.add(str(m.group(1)))

    for key_name in sorted(key_set):
        shape = _collect_smallest_linear_weight_dims(
            sd, rf"^obs_encoder\._per_key_mlps\.{re.escape(key_name)}\.net\.(\d+)\.weight$"
        )
        if shape is not None:
            per_key_first_dims[key_name] = shape

        per_key_linear_counts[key_name] = _count_linear_layers_under(
            sd, rf"^obs_encoder\._per_key_mlps\.{re.escape(key_name)}\.net\.(\d+)\.weight$"
        )
        per_key_layernorm[key_name] = _has_layernorm_weights_under(
            sd, rf"^obs_encoder\._per_key_mlps\.{re.escape(key_name)}\.net\.(\d+)\.weight$"
        )

    if len(per_key_first_dims) > 0:
        raw_states_dict = states_dict
        if raw_states_dict is None:
            raise RuntimeError(
                "Checkpoint indicates a dict/per-key obs encoder, but expert npz has no states_dict."
            )

        horizon_candidates: List[int] = []
        for key_name, (_out_dim, in_dim_total) in per_key_first_dims.items():
            if key_name not in raw_states_dict:
                continue
            raw_dim = int(raw_states_dict[key_name].shape[1])
            if raw_dim <= 0:
                continue
            if in_dim_total % raw_dim != 0:
                raise RuntimeError(
                    f"Cannot infer obs_horizon for key '{key_name}': "
                    f"checkpoint expects input dim {in_dim_total}, raw expert dim is {raw_dim}."
                )
            horizon_candidates.append(in_dim_total // raw_dim)

        if len(horizon_candidates) == 0:
            obs_horizon = 1
        else:
            first_h = int(horizon_candidates[0])
            if any(int(h) != first_h for h in horizon_candidates):
                raise RuntimeError(
                    f"Inconsistent inferred obs_horizon from per-key encoder: {horizon_candidates}"
                )
            obs_horizon = first_h

        sample_key = sorted(per_key_first_dims.keys())[0]
        per_key_hidden = int(per_key_first_dims[sample_key][0])
        per_key_layers = int(per_key_linear_counts[sample_key])
        encoder_layernorm = bool(per_key_layernorm[sample_key])

        use_goal_delta = "goal_delta" in per_key_first_dims

        agg = "sum"
        concat_proj_dim = None
        for k, v in sd.items():
            if "concat_proj" in k and k.endswith(".weight") and v.ndim == 2:
                agg = "concat"
                concat_proj_dim = int(v.shape[0])
                break

        return {
            "input_mode": "dict",
            "obs_horizon": int(obs_horizon),
            "encoder_kind": "dict",
            "per_key_hidden": per_key_hidden,
            "per_key_layers": per_key_layers,
            "encoder_layernorm": encoder_layernorm,
            "use_goal_delta": use_goal_delta,
            "agg": agg,
            "concat_proj_dim": concat_proj_dim,
        }

    flat_shape = _collect_smallest_linear_weight_dims(
        sd, r"^obs_encoder\._flat_mlp\.net\.(\d+)\.weight$"
    )
    if flat_shape is not None:
        flat_hidden, flat_in_total = flat_shape
        raw_dim = int(states.shape[1])
        if raw_dim <= 0:
            obs_horizon = 1
        else:
            if flat_in_total % raw_dim != 0:
                raise RuntimeError(
                    f"Cannot infer obs_horizon for flat encoder: "
                    f"checkpoint expects input dim {flat_in_total}, raw expert dim is {raw_dim}."
                )
            obs_horizon = flat_in_total // raw_dim

        flat_layers = _count_linear_layers_under(sd, r"^obs_encoder\._flat_mlp\.net\.(\d+)\.weight$")
        encoder_layernorm = _has_layernorm_weights_under(sd, r"^obs_encoder\._flat_mlp\.net\.(\d+)\.weight$")

        return {
            "input_mode": "flat",
            "obs_horizon": int(obs_horizon),
            "encoder_kind": "flat",
            "per_key_hidden": int(flat_hidden),
            "per_key_layers": int(flat_layers),
            "encoder_layernorm": bool(encoder_layernorm),
            "use_goal_delta": False,
            "agg": "sum",
            "concat_proj_dim": None,
        }

    return {
        "input_mode": "flat",
        "obs_horizon": 1,
        "encoder_kind": "flat",
        "per_key_hidden": 256,
        "per_key_layers": 2,
        "encoder_layernorm": False,
        "use_goal_delta": False,
        "agg": "sum",
        "concat_proj_dim": None,
    }


def _build_example_obs_from_expert_buf(
    expert_buf: Dict[str, Any],
    input_mode: str,
    obs_horizon: int,
    device: torch.device,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Build a shape-correct example obs so lazy encoder modules are materialized correctly.
    """
    if input_mode == "dict":
        states_dict = expert_buf.get("states_dict", None)
        if states_dict is None:
            raise RuntimeError("Need dict expert data to build dict example obs.")

        out: Dict[str, torch.Tensor] = {}
        for k, v in states_dict.items():
            base = np.asarray(v[:1], dtype=np.float32)
            if obs_horizon > 1:
                rep = np.repeat(base[:, None, :], repeats=obs_horizon, axis=1)
                out[k] = torch.as_tensor(rep, dtype=torch.float32, device=device)
            else:
                out[k] = torch.as_tensor(base, dtype=torch.float32, device=device)
        return out

    if input_mode == "flat":
        base = np.asarray(expert_buf["states"][:1], dtype=np.float32)
        if obs_horizon > 1:
            rep = np.repeat(base[:, None, :], repeats=obs_horizon, axis=1)
            return torch.as_tensor(rep, dtype=torch.float32, device=device)
        return torch.as_tensor(base, dtype=torch.float32, device=device)

    raise ValueError(f"Unknown input_mode: {input_mode}")


def _warmup_reward_model(
    model: nn.Module,
    example_obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
    action_dim: int,
    device: torch.device,
) -> None:
    include_action = bool(getattr(model, "include_action", True))
    model.eval()
    with torch.no_grad():
        if isinstance(example_obs, dict):
            batch_size = next(iter(example_obs.values())).shape[0]
            dummy_obs = {k: torch.zeros_like(v, device=device) for k, v in example_obs.items()}
        else:
            batch_size = example_obs.shape[0]
            dummy_obs = torch.zeros_like(example_obs, device=device)

        if include_action:
            dummy_act = torch.zeros((batch_size, action_dim), dtype=torch.float32, device=device)
        else:
            dummy_act = None

        _ = model(dummy_obs, dummy_act)


def _try_build_reward_from_state_dict(
    sd: Mapping[str, torch.Tensor],
    expert_buf: Dict[str, Any],
    device: torch.device,
    cfg_hints: Dict[str, Any],
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Reconstruct the reward network directly from state_dict shapes.
    """
    action_dim_from_data = int(expert_buf["actions"].shape[1])

    reward_head_candidates = _parse_top_level_reward_head(sd, action_dim_from_data=action_dim_from_data)
    encoder_meta = _infer_obs_encoder_meta(sd, expert_buf=expert_buf)

    build_errors: List[str] = []

    for cand in reward_head_candidates:
        try:
            state_embed_dim = int(cand["state_embed_dim"])
            include_action = bool(cand["include_action"])
            action_dim = int(cand["action_dim"])

            reward_act = str(cfg_hints.get("reward_act", "silu"))
            reward_layernorm = (
                bool(cfg_hints["reward_layernorm"])
                if cfg_hints.get("reward_layernorm", None) is not None
                else bool(cand["layernorm"])
            )

            encoder_act = str(cfg_hints.get("encoder_act", "silu"))
            encoder_layernorm = (
                bool(cfg_hints["encoder_layernorm"])
                if cfg_hints.get("encoder_layernorm", None) is not None
                else bool(encoder_meta["encoder_layernorm"])
            )
            encoder_agg = (
                str(cfg_hints["encoder_agg"]).lower()
                if cfg_hints.get("encoder_agg", None) is not None
                else str(encoder_meta["agg"])
            )
            encoder_use_goal_delta = (
                bool(cfg_hints["encoder_use_goal_delta"])
                if cfg_hints.get("encoder_use_goal_delta", None) is not None
                else bool(encoder_meta["use_goal_delta"])
            )
            encoder_concat_proj_dim = (
                int(cfg_hints["encoder_concat_proj_dim"])
                if cfg_hints.get("encoder_concat_proj_dim", None) is not None
                else (
                    int(encoder_meta["concat_proj_dim"])
                    if encoder_meta["concat_proj_dim"] is not None
                    else state_embed_dim
                )
            )

            obs_encoder = ObsEncoder(
                out_dim=state_embed_dim,
                per_key_hidden=int(encoder_meta["per_key_hidden"]),
                per_key_layers=int(encoder_meta["per_key_layers"]),
                agg=encoder_agg,
                concat_proj_dim=encoder_concat_proj_dim,
                act=encoder_act,
                layernorm=encoder_layernorm,
                use_goal_delta=encoder_use_goal_delta,
            )

            model = RewardNetwork(
                state_embed_dim=state_embed_dim,
                include_action=include_action,
                action_dim=action_dim,
                hidden=int(cand["hidden"]),
                layers=int(cand["layers"]),
                act=reward_act,
                layernorm=reward_layernorm,
                obs_encoder=obs_encoder,
            ).to(device)

            example_obs = _build_example_obs_from_expert_buf(
                expert_buf=expert_buf,
                input_mode=str(encoder_meta["input_mode"]),
                obs_horizon=int(encoder_meta["obs_horizon"]),
                device=device,
            )
            _warmup_reward_model(model, example_obs=example_obs, action_dim=action_dim_from_data, device=device)

            model.load_state_dict(sd, strict=True)
            model.eval()

            runtime_meta = {
                "input_mode": str(encoder_meta["input_mode"]),
                "obs_horizon": int(encoder_meta["obs_horizon"]),
            }
            return model, runtime_meta

        except Exception as e:
            build_errors.append(
                f"[candidate={cand} | encoder_meta={encoder_meta}] {type(e).__name__}: {e}"
            )

    raise RuntimeError(
        "Failed to reconstruct RewardNetwork from state_dict.\n"
        "Collected errors:\n  - " + "\n  - ".join(build_errors)
    )


def _load_reward_model(
    reward_path: Union[str, Path],
    device: torch.device,
    expert_buf: Dict[str, Any],
    cfg_path: Optional[str] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load reward model from:
      - full nn.Module checkpoint, or
      - state_dict checkpoint
    """
    cfg_file = _find_reward_cfg_file(reward_path, explicit_cfg_path=cfg_path)
    cfg_hints = _load_cfg_hints(cfg_file)

    obj = _load_torch_object(reward_path, device=device)

    if isinstance(obj, nn.Module):
        model = obj.to(device)
        model.eval()

        sd = _strip_common_prefix_from_state_dict(model.state_dict())
        encoder_meta = _infer_obs_encoder_meta(sd, expert_buf=expert_buf)
        runtime_meta = {
            "input_mode": str(encoder_meta["input_mode"]),
            "obs_horizon": int(encoder_meta["obs_horizon"]),
        }
        return model, runtime_meta

    sd = _extract_state_dict_from_checkpoint_obj(obj)
    if sd is None:
        raise RuntimeError(
            "Unsupported checkpoint format. Expected either:\n"
            "  1) a full nn.Module saved by torch.save(model, ...), or\n"
            "  2) a state_dict (possibly wrapped in a checkpoint dict)."
        )

    model, runtime_meta = _try_build_reward_from_state_dict(
        sd=sd,
        expert_buf=expert_buf,
        device=device,
        cfg_hints=cfg_hints,
    )
    return model, runtime_meta


# Scoring

def _score_trajectory(
    model: nn.Module,
    raw_obs_seq: Union[np.ndarray, Dict[str, np.ndarray]],
    act_seq: np.ndarray,
    device: torch.device,
    input_mode: str,
    obs_horizon: int,
    batch_size: int = 4096,
) -> float:
    """
    Sum learned rewards over one trajectory.
    Handles sliding-window obs if obs_horizon > 1.
    """
    num_steps = int(act_seq.shape[0])
    if num_steps <= 0:
        return 0.0

    obs_seq = _window_obs_seq(raw_obs_seq, input_mode=input_mode, obs_horizon=obs_horizon)
    include_action = bool(getattr(model, "include_action", True))
    total = 0.0

    with torch.no_grad():
        for lo in range(0, num_steps, batch_size):
            hi = min(lo + batch_size, num_steps)

            obs_b = _slice_obs_seq(obs_seq, lo, hi)
            obs_t = _obs_to_torch(obs_b, device)

            if include_action:
                act_t = torch.as_tensor(act_seq[lo:hi], dtype=torch.float32, device=device)
            else:
                act_t = None

            r = model(obs_t, act_t)
            total += float(r.sum().item())

    return total


# Bad trajectory generation

def _sample_bad_action(
    action_space: gym.Space,
    rng: np.random.Generator,
    bad_policy: str = "random",
) -> Tuple[Any, np.ndarray]:
    """
    Returns:
      - action_env: sent to env.step(...)
      - action_model: flattened float32 action for the reward model
    """
    if isinstance(action_space, gym.spaces.Discrete):
        if bad_policy == "zero":
            a_int = 0
        else:
            a_int = int(rng.integers(action_space.n))

        one_hot = np.zeros(action_space.n, dtype=np.float32)
        one_hot[a_int] = 1.0
        return a_int, one_hot

    if isinstance(action_space, gym.spaces.Box):
        if bad_policy == "zero":
            a_env = np.zeros(action_space.shape, dtype=action_space.dtype)
            a_env = np.clip(a_env, action_space.low, action_space.high)
        else:
            low = np.asarray(action_space.low)
            high = np.asarray(action_space.high)
            if np.all(np.isfinite(low)) and np.all(np.isfinite(high)):
                a_env = rng.uniform(low=low, high=high).astype(action_space.dtype)
            else:
                a_env = np.asarray(action_space.sample(), dtype=action_space.dtype)

        a_model = np.asarray(a_env, dtype=np.float32).reshape(-1)
        return a_env, a_model

    a_env = action_space.sample()
    a_model = np.asarray(a_env, dtype=np.float32).reshape(-1)
    return a_env, a_model


def _rollout_full_bad_episode(
    env: gym.Env,
    rng: np.random.Generator,
    bad_policy: str,
    input_mode: str,
    obs_keys: Optional[Tuple[str, ...]],
) -> Tuple[Union[np.ndarray, Dict[str, np.ndarray]], np.ndarray, np.ndarray, int, float]:
    """
    Roll out ONE complete poor trajectory until env termination/truncation.

    Returns:
      - raw_obs_seq: prepared observations for reward-model scoring
      - act_seq: flattened actions for reward-model scoring
      - env_reward_seq: true environment rewards at every step
      - length: number of generated steps
      - env_return: cumulative true environment reward over the full rollout
    """
    seed_val = int(rng.integers(0, 2**31 - 1))
    obs_raw, _ = env.reset(seed=seed_val)

    obs_steps: List[Union[np.ndarray, Dict[str, np.ndarray]]] = []
    act_steps: List[np.ndarray] = []
    env_reward_steps: List[float] = []

    done = False
    while not done:
        obs_prepared = _prepare_single_obs_for_mode(obs_raw, input_mode=input_mode, obs_keys=obs_keys)
        action_env, action_model = _sample_bad_action(env.action_space, rng=rng, bad_policy=bad_policy)

        obs_steps.append(obs_prepared)
        act_steps.append(np.asarray(action_model, dtype=np.float32).reshape(-1))

        obs_next, reward_env, terminated, truncated, _info = env.step(action_env)
        env_reward_steps.append(float(reward_env))
        obs_raw = obs_next
        done = bool(terminated or truncated)

    raw_obs_seq = _stack_obs_list(obs_steps, input_mode=input_mode, obs_keys=obs_keys)
    act_seq = np.stack(act_steps, axis=0).astype(np.float32)
    env_reward_seq = np.asarray(env_reward_steps, dtype=np.float32)
    env_return = float(env_reward_seq.sum())

    return raw_obs_seq, act_seq, env_reward_seq, int(act_seq.shape[0]), env_return


def _trim_obs_seq(
    obs_seq: Union[np.ndarray, Dict[str, np.ndarray]],
    target_len: int,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    if isinstance(obs_seq, dict):
        return {k: v[:target_len].copy() for k, v in obs_seq.items()}
    return obs_seq[:target_len].copy()


def _generate_bad_rollout(
    env: gym.Env,
    horizon: int,
    model: nn.Module,
    device: torch.device,
    input_mode: str,
    obs_keys: Optional[Tuple[str, ...]],
    obs_horizon: int,
    rng: np.random.Generator,
    bad_policy: str,
    batch_size: int,
    max_resample_tries: int = 200,
) -> Tuple[float, int]:
    """
    Generate and score one bad/generated trajectory.

    Non-Pen environments:
      - Keep the previous behavior.
      - Generate a full bad trajectory.
      - Require full_len >= expert trajectory length.
      - Trim the generated trajectory to the expert trajectory length before scoring.

    Pen / AdroitHandPen environments:
      - Do NOT force generated length to match expert length.
      - Do NOT require generated length >= expert length.
      - Use the generated trajectory's own natural episode length.
      - Accept it only if its true environment cumulative reward is <= -500.
    """
    target_horizon = max(1, int(horizon))
    pen_env = _is_pen_env(env)

    for _ in range(max_resample_tries):
        raw_obs_full, act_full, env_reward_full, full_len, full_env_return = _rollout_full_bad_episode(
            env=env,
            rng=rng,
            bad_policy=bad_policy,
            input_mode=input_mode,
            obs_keys=obs_keys,
        )

        if pen_env:
            # For Pen, keep the generated episode at its own natural length.
            # The only extra constraint is that the generated trajectory's true
            # cumulative environment reward must be no higher than -500.
            if full_env_return > PEN_GENERATED_MAX_CUMULATIVE_REWARD:
                continue

            total = _score_trajectory(
                model=model,
                raw_obs_seq=raw_obs_full,
                act_seq=act_full,
                device=device,
                input_mode=input_mode,
                obs_horizon=obs_horizon,
                batch_size=batch_size,
            )

            return total, int(full_len)

        # For non-Pen environments, keep the old length-matching behavior.
        if full_len < target_horizon:
            continue

        raw_obs_trim = _trim_obs_seq(raw_obs_full, target_horizon)
        act_trim = act_full[:target_horizon].copy()

        total = _score_trajectory(
            model=model,
            raw_obs_seq=raw_obs_trim,
            act_seq=act_trim,
            device=device,
            input_mode=input_mode,
            obs_horizon=obs_horizon,
            batch_size=batch_size,
        )

        return total, int(act_trim.shape[0])

    if pen_env:
        raise RuntimeError(
            f"Failed to sample a Pen generated trajectory with cumulative env reward <= "
            f"{PEN_GENERATED_MAX_CUMULATIVE_REWARD} after {max_resample_tries} tries. "
            f"For Pen, generated length is not matched to expert length; it uses its natural episode length."
        )

    raise RuntimeError(
        f"Failed to sample a generated trajectory with length >= target expert length "
        f"{target_horizon} after {max_resample_tries} tries."
    )


# Main

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a learned reward network by ranking expert vs poor trajectories."
    )
    parser.add_argument(
        "--reward_path",
        type=str,
        default=r"results/AdroitHandPen-v1/r_net.pt",
        help="Path to r_net.pt, e.g. results/PointMaze_UMaze-v3/20260209_102132/r_net.pt",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default=None,
        help="Optional path to a YAML config containing r_net settings. "
             "If omitted, the script will try to auto-find one near reward_path.",
    )
    parser.add_argument(
        "--num_traj",
        type=int,
        default=200,
        help="Number of expert trajectories and generated trajectories each.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=52,
        help="Random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="cpu / cuda / auto",
    )
    parser.add_argument(
        "--env_id",
        type=str,
        default=None,
        help="Optional explicit env id override if the log folder name is not directly usable as a gym id.",
    )
    parser.add_argument(
        "--bad_policy",
        type=str,
        default="random",
        choices=["random", "zero"],
        help="How to generate poor trajectories.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4096,
        help="Batch size used when scoring trajectories.",
    )
    parser.add_argument(
        "--max_resample_tries",
        type=int,
        default=200,
        help="Maximum retries for generated trajectory sampling. Non-Pen requires enough length; Pen requires cumulative env reward <= -500.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    reward_path = Path(args.reward_path).resolve()
    if not reward_path.exists():
        raise FileNotFoundError(f"Reward checkpoint not found: {reward_path}")

    env_tag = _infer_env_tag_from_reward_path(reward_path)
    expert_npz = _find_expert_npz(env_tag, env_id_override=args.env_id)
    expert_buf = _load_expert_buffer(expert_npz)

    episodes: List[Tuple[int, int]] = expert_buf["episodes"]
    num_expert_available = len(episodes)
    if num_expert_available <= 0:
        raise RuntimeError("The expert trajectory archive contains no episodes.")

    env_candidates = _candidate_env_ids_from_tag(env_tag, env_id_override=args.env_id)
    env, resolved_env_id = _make_env_from_candidates(env_candidates)
    pen_env = _is_pen_env(env)

    model, runtime_meta = _load_reward_model(
        reward_path=reward_path,
        device=device,
        expert_buf=expert_buf,
        cfg_path=args.cfg,
    )

    input_mode = str(runtime_meta["input_mode"])
    obs_horizon = int(runtime_meta["obs_horizon"])

    obs_keys: Optional[Tuple[str, ...]] = expert_buf["obs_keys"]
    states = expert_buf["states"]
    actions = expert_buf["actions"]
    states_dict = expert_buf["states_dict"]

    rng = np.random.default_rng(args.seed)

    chosen_ep_ids = _sample_episode_ids(num_expert_available, args.num_traj, rng=rng)

    expert_scores: List[float] = []
    expert_lengths: List[int] = []

    for ep_id in chosen_ep_ids:
        s, e = episodes[int(ep_id)]
        traj_len = int(e - s + 1)

        if input_mode == "dict":
            if states_dict is None:
                raise RuntimeError("Model expects dict observations, but expert npz does not contain states_dict.")
            raw_obs_seq = {k: v[s:e + 1].astype(np.float32) for k, v in states_dict.items()}
        else:
            raw_obs_seq = states[s:e + 1].astype(np.float32)

        act_seq = actions[s:e + 1].astype(np.float32)

        total = _score_trajectory(
            model=model,
            raw_obs_seq=raw_obs_seq,
            act_seq=act_seq,
            device=device,
            input_mode=input_mode,
            obs_horizon=obs_horizon,
            batch_size=args.batch_size,
        )

        expert_scores.append(total)
        expert_lengths.append(traj_len)

    generated_scores: List[float] = []
    generated_lengths: List[int] = []

    for horizon in expert_lengths:
        total, realized_len = _generate_bad_rollout(
            env=env,
            horizon=horizon,
            model=model,
            device=device,
            input_mode=input_mode,
            obs_keys=obs_keys,
            obs_horizon=obs_horizon,
            rng=rng,
            bad_policy=args.bad_policy,
            batch_size=args.batch_size,
            max_resample_tries=args.max_resample_tries,
        )
        generated_scores.append(total)
        generated_lengths.append(realized_len)

    try:
        env.close()
    except Exception:
        pass

    if len(expert_lengths) != len(generated_lengths):
        raise RuntimeError(
            f"Length list size mismatch: len(expert_lengths)={len(expert_lengths)}, "
            f"len(generated_lengths)={len(generated_lengths)}"
        )

    mismatch_pairs = [
        (i, el, gl)
        for i, (el, gl) in enumerate(zip(expert_lengths, generated_lengths))
        if int(el) != int(gl)
    ]

    if (not pen_env) and len(mismatch_pairs) > 0:
        preview = mismatch_pairs[:10]
        raise RuntimeError(
            "Generated trajectories were not trimmed to expert lengths correctly. "
            f"First mismatches: {preview}"
        )

    if pen_env and len(mismatch_pairs) > 0:
        print(
            "[Pen] Length matching is intentionally disabled. "
            f"Expert/generated length mismatches are allowed. First mismatches: {mismatch_pairs[:10]}"
        )

    k = int(args.num_traj)
    scores = np.asarray(expert_scores + generated_scores, dtype=np.float64)
    gt = np.asarray([1] * k + [0] * k, dtype=np.int64)

    order = np.argsort(-scores, kind="mergesort")
    pred = np.zeros_like(gt)
    pred[order[:k]] = 1

    overlap_rate = float((pred == gt).mean())

    tp = int(((pred == 1) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())

    print("=" * 80)
    print("Reward ranking evaluation")
    print("=" * 80)
    print(f"Reward checkpoint : {reward_path}")
    print(f"Inferred env tag  : {env_tag}")
    print(f"Resolved env id   : {resolved_env_id}")
    print(f"Env candidates    : {env_candidates}")
    print(f"Expert npz        : {expert_npz}")
    print(f"Device            : {device}")
    print(f"Model input mode  : {input_mode}")
    print(f"Obs horizon       : {obs_horizon}")
    print(f"Bad policy        : {args.bad_policy}")
    print(f"Pen env detected  : {pen_env}")
    if pen_env:
        print(f"Pen gen reward cap: cumulative env reward <= {PEN_GENERATED_MAX_CUMULATIVE_REWARD}")
    print(f"Expert available  : {num_expert_available}")
    print(f"Expert sampled    : {k}")
    print(f"Generated sampled : {k}")
    print("-" * 80)
    print(f"Expert length mean: {np.mean(expert_lengths):.6f}")
    print(f"Gen length mean   : {np.mean(generated_lengths):.6f}")
    print(f"Length aligned    : {all(int(a) == int(b) for a, b in zip(expert_lengths, generated_lengths)) if not pen_env else 'skipped for Pen'}")
    print("-" * 80)
    print(f"Expert score mean : {np.mean(expert_scores):.6f}")
    print(f"Expert score std  : {np.std(expert_scores):.6f}")
    print(f"Gen score mean    : {np.mean(generated_scores):.6f}")
    print(f"Gen score std     : {np.std(generated_scores):.6f}")
    print("-" * 80)
    print(f"TP                : {tp}")
    print(f"TN                : {tn}")
    print(f"FP                : {fp}")
    print(f"FN                : {fn}")
    print(f"Overlap rate      : {overlap_rate:.6f}  ({overlap_rate * 100.0:.2f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
