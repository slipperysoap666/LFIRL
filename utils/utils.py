from typing import Dict, Union, List, Optional, Tuple, Any, Mapping, Set
import random
import numpy as np
import torch
import gym_pusht
import gymnasium as gym
from gymnasium import spaces

FLAT_KEY_SEP = "__"

# Kitchen (Minari D4RL/kitchen/complete-v2) task set
KITCHEN_MINARI_COMPLETE_V2_TASKS: List[str] = [
    "kettle",
    "light switch",
    "microwave",
    "slide cabinet",
]



def seed_everything(seed: int) -> None:
    """Set global RNG seeds for Python, NumPy, and PyTorch (CPU/CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# Franka-Kitchen nested goal Dict obs -> vector goal (SB3-friendly)

def _is_nested_goal_dict_obs_space(obs_space: spaces.Space) -> bool:
    """
    Franka-Kitchen observation space example:
      Dict(
        'achieved_goal': Dict('kettle': Box(..., (7,)), ...),
        'desired_goal' : Dict(...),
        'observation'  : Box(..., (59,))
      )
    """
    if not isinstance(obs_space, spaces.Dict):
        return False
    if "achieved_goal" not in obs_space.spaces or "desired_goal" not in obs_space.spaces:
        return False
    return isinstance(obs_space.spaces["achieved_goal"], spaces.Dict) or isinstance(obs_space.spaces["desired_goal"], spaces.Dict)


def _sorted_goal_keys_from_space(goal_space: spaces.Space) -> List[str]:
    if isinstance(goal_space, spaces.Dict):
        return sorted(list(goal_space.spaces.keys()))
    return []


def _sum_goal_dim(goal_space: spaces.Space, keys: List[str]) -> int:
    if not isinstance(goal_space, spaces.Dict):
        return int(np.prod(getattr(goal_space, "shape", (0,))))
    total = 0
    for k in keys:
        sp = goal_space.spaces[k]
        total += int(np.prod(sp.shape))
    return total


def _concat_goal_leaf_np(goal: Any, goal_keys: List[str]) -> np.ndarray:
    """
    goal: dict[str, array] where each leaf is shape (d,) or (..., d)
    return: concatenated vector shape (sum_d,)
    """
    assert isinstance(goal, dict), "Expected nested goal dict."
    parts = []
    for k in goal_keys:
        if k not in goal:
            raise KeyError(
                f"Kitchen goal dict missing key '{k}'. Available keys={sorted(list(goal.keys()))}."
            )
        v = goal[k]
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        parts.append(v)
    if len(parts) == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(parts, axis=0).astype(np.float32, copy=False)


class KitchenNestedGoalToVecWrapper(gym.ObservationWrapper):
    """
    Convert Franka-Kitchen nested goal dict into flat vectors:
      achieved_goal: Dict(...) -> Box(sum_d,)
      desired_goal : Dict(...) -> Box(sum_d,)
    Keep top-level keys as Dict for SB3 MultiInputPolicy compatibility.

    If tasks_to_keep is provided, only those tasks are concatenated.
    This is used to align evaluation env with Minari datasets like complete-v2.
    """

    def __init__(self, env: gym.Env, tasks_to_keep: Optional[List[str]] = None):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Dict), "Expected Dict observation space."
        obs_space: spaces.Dict = env.observation_space

        self._tasks_to_keep = list(tasks_to_keep) if tasks_to_keep is not None else None
        keep_set = set(self._tasks_to_keep) if self._tasks_to_keep is not None else None

        self._goal_keys_ach = _sorted_goal_keys_from_space(obs_space.spaces.get("achieved_goal"))
        self._goal_keys_des = _sorted_goal_keys_from_space(obs_space.spaces.get("desired_goal"))

        # Filter to the dataset task set if requested
        if keep_set is not None:
            self._goal_keys_ach = [k for k in self._goal_keys_ach if k in keep_set]
            self._goal_keys_des = [k for k in self._goal_keys_des if k in keep_set]

        ach_space = obs_space.spaces.get("achieved_goal")
        des_space = obs_space.spaces.get("desired_goal")
        obs_box = obs_space.spaces.get("observation")

        ach_dim = _sum_goal_dim(ach_space, self._goal_keys_ach)
        des_dim = _sum_goal_dim(des_space, self._goal_keys_des)

        new_spaces = dict(obs_space.spaces)

        # observation stays Box; cast dtype for consistency
        if isinstance(obs_box, spaces.Box):
            new_spaces["observation"] = spaces.Box(
                low=np.asarray(obs_box.low, dtype=np.float32),
                high=np.asarray(obs_box.high, dtype=np.float32),
                shape=obs_box.shape,
                dtype=np.float32,
            )

        # achieved_goal / desired_goal become Box vectors
        if isinstance(ach_space, spaces.Dict):
            new_spaces["achieved_goal"] = spaces.Box(low=-np.inf, high=np.inf, shape=(ach_dim,), dtype=np.float32)
        if isinstance(des_space, spaces.Dict):
            new_spaces["desired_goal"] = spaces.Box(low=-np.inf, high=np.inf, shape=(des_dim,), dtype=np.float32)

        self.observation_space = spaces.Dict(new_spaces)

    def observation(self, obs: Any):
        if not isinstance(obs, dict):
            return obs

        out = dict(obs)

        if "observation" in out:
            out["observation"] = np.asarray(out["observation"], dtype=np.float32)

        if "achieved_goal" in out and isinstance(out["achieved_goal"], dict):
            out["achieved_goal"] = _concat_goal_leaf_np(out["achieved_goal"], self._goal_keys_ach)
        if "desired_goal" in out and isinstance(out["desired_goal"], dict):
            out["desired_goal"] = _concat_goal_leaf_np(out["desired_goal"], self._goal_keys_des)

        return out


def _maybe_wrap_franka_kitchen_obs(env: gym.Env, tasks_to_keep: Optional[List[str]] = None) -> gym.Env:
    """
    Make Kitchen nested-goal dict obs SB3-friendly by flattening nested dict goals.
    Safe no-op for non-Kitchen envs.
    """
    try:
        if _is_nested_goal_dict_obs_space(env.observation_space):
            if tasks_to_keep is None:
                print("[Env] Detected nested goal Dict obs. Wrapping env (KitchenNestedGoalToVecWrapper).")
            else:
                print(f"[Env] Detected nested goal Dict obs. Wrapping env (KitchenNestedGoalToVecWrapper, tasks_to_keep={tasks_to_keep}).")
            return KitchenNestedGoalToVecWrapper(env, tasks_to_keep=tasks_to_keep)
    except Exception:
        pass
    return env



def _flatten_nested_dict_leaves(
    d: Mapping[str, Any],
    prefix: str = "",
    sep: str = FLAT_KEY_SEP,
) -> Dict[str, np.ndarray]:
    """
    Deterministically flatten a (possibly nested) mapping into leaf arrays with keys joined by sep.
    """
    out: Dict[str, np.ndarray] = {}
    for k in sorted(d.keys()):
        v = d[k]
        kk = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_nested_dict_leaves(v, prefix=kk, sep=sep))
        else:
            out[kk] = np.asarray(v)
    return out


def canonicalize_obs(obs: Any) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Convert env observation into one of:
      - np.ndarray (flat obs)
      - Dict[str, np.ndarray] where all values are 1D float32 arrays (leaf dict obs)
    """
    if isinstance(obs, dict):
        leaf = _flatten_nested_dict_leaves(obs, sep=FLAT_KEY_SEP)
        return {k: np.asarray(v, dtype=np.float32).reshape(-1) for k, v in leaf.items()}
    return np.asarray(obs, dtype=np.float32)


def _concat_goal_vec(goal_like: Any) -> Optional[np.ndarray]:
    """
    Turn achieved_goal/desired_goal (array or nested dict) into a single 1D vector.
    Used only for fallback success metric when info['success'] is missing.
    """
    if goal_like is None:
        return None
    if isinstance(goal_like, dict):
        leaf = _flatten_nested_dict_leaves(goal_like, sep=FLAT_KEY_SEP)
        keys = sorted(leaf.keys())
        parts = [np.asarray(leaf[k], dtype=np.float32).reshape(-1) for k in keys]
        if len(parts) == 0:
            return None
        return np.concatenate(parts, axis=0)
    return np.asarray(goal_like, dtype=np.float32).reshape(-1)



def _build_obs_window_dict(buf: List[Dict[str, np.ndarray]], device) -> Dict[str, torch.Tensor]:
    """List of dict observations (len=To) -> Dict[str, Tensor [1, To, Sk]]."""
    keys = sorted(buf[0].keys())
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        seq = [o[k] for o in buf]
        arr = np.stack(seq, axis=0)  # [To, Sk]
        out[k] = torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)
    return out


def _build_obs_window_tensor(buf: List[np.ndarray], device) -> torch.Tensor:
    """List of flat arrays (len=To) -> Tensor [1, To, S]."""
    arr = np.stack(buf, axis=0)  # [To, S]
    return torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)



def to_device_obs(
    obs: Union[torch.Tensor, Dict[str, Any]],
    device,
) -> Union[torch.Tensor, Dict[str, Any]]:
    """Move an observation (Tensor or (possibly nested) Dict) onto a device."""
    if isinstance(obs, dict):
        out = {}
        for k, v in obs.items():
            if isinstance(v, dict):
                out[k] = to_device_obs(v, device)
            else:
                out[k] = v.to(device) if isinstance(v, torch.Tensor) else torch.as_tensor(v, dtype=torch.float32, device=device)
        return out
    return obs.to(device)



def compute_obs_stats_flat(
    states: np.ndarray,
    indices: Optional[np.ndarray] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute mean/std for flat observations, optionally on a subset of indices."""
    x = states if indices is None else states[indices]
    mean = torch.tensor(x.mean(axis=0), dtype=torch.float32)
    std = torch.tensor(x.std(axis=0) + 1e-6, dtype=torch.float32)
    return mean, std


def compute_obs_stats_dict(
    states_dict: Dict[str, np.ndarray],
    indices: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Compute per-key mean/std for dict observations, optionally on a subset of indices."""
    mean_d: Dict[str, torch.Tensor] = {}
    std_d: Dict[str, torch.Tensor] = {}
    for k, arr in states_dict.items():
        x = arr if indices is None else arr[indices]
        mean_d[k] = torch.tensor(x.mean(axis=0), dtype=torch.float32)
        std_d[k] = torch.tensor(x.std(axis=0) + 1e-6, dtype=torch.float32)
    return mean_d, std_d



def _infer_env_id_from_env(env: gym.Env) -> str:
    """Best-effort env_id inference from env.spec.id."""
    try:
        spec = getattr(env, "spec", None)
        if spec is None and hasattr(env, "unwrapped"):
            spec = getattr(env.unwrapped, "spec", None)
        if spec is not None and getattr(spec, "id", None) is not None:
            return str(spec.id)
    except Exception:
        pass
    return ""


def _is_kitchen_env_id(env_id: str) -> bool:
    s = str(env_id).lower()
    return ("kitchen" in s) or ("frankakitchen" in s) or ("franka_kitchen" in s)


def _is_pusht_env_id(env_id: str) -> bool:
    s = str(env_id).lower()
    # Common id: "gym_pusht/PushT-v0"
    return ("pusht" in s) or ("gym_pusht" in s)


def _is_robotics_like_env_id(env_id: str) -> bool:
    """
    Envs that typically come from gymnasium-robotics (need gymnasium_robotics + mujoco).
    """
    s = str(env_id).lower()
    return (
        ("antmaze" in s)
        or ("pointmaze" in s)
        or ("maze" in s)
        or ("kitchen" in s)
        or ("franka" in s)
        or ("pen" in s)
        or ("gymnasium_robotics" in s)
    )

def _is_adroit_hand_pen_env_id(env_id) -> bool:
    if env_id is None:
        return False
    s = str(env_id).lower()
    return ("adroit" in s and "pen" in s) or ("handpen" in s)


def _kitchen_make_kwargs_minari_complete_v2() -> Dict[str, Any]:
    """Match Minari D4RL/kitchen/complete-v2 environment kwargs."""
    return dict(
        remove_task_when_completed=False,
        terminate_on_tasks_completed=False,
        tasks_to_complete=list(KITCHEN_MINARI_COMPLETE_V2_TASKS),
    )


def _kitchen_update_completed_tasks_from_info(
    completed: Set[str],
    info: Any,
) -> Tuple[Set[str], bool]:
    """
    Update `completed` from kitchen info dict.
    Returns (completed, saw_task_info_flag).
    """
    saw = False
    if not isinstance(info, dict):
        return completed, saw

    if "episode_task_completions" in info:
        try:
            v = info["episode_task_completions"]
            if v is None:
                pass
            else:
                if isinstance(v, (set, tuple, list)):
                    completed = set([str(x) for x in v])
                else:
                    completed = set([str(v)])
            saw = True
        except Exception:
            pass
        return completed, saw

    if "step_task_completions" in info:
        try:
            v = info["step_task_completions"]
            if v is None:
                pass
            else:
                if isinstance(v, (set, tuple, list)):
                    completed |= set([str(x) for x in v])
                else:
                    completed.add(str(v))
            saw = True
        except Exception:
            pass

    return completed, saw



def make_env(env_id: str, seed: Optional[int] = None) -> gym.Env:
    """
    Create a Gymnasium env.

    IMPORTANT:
      - Only register gymnasium_robotics when needed (maze/kitchen/etc).
      - PushT (gym_pusht/PushT-v0) does NOT require gymnasium_robotics.
      - For FrankaKitchen, align evaluation env with Minari D4RL/kitchen/complete-v2 tasks.
    """
    env_id_clean = str(env_id).strip()

    # Register gymnasium_robotics only for robotics-like environments
    if _is_robotics_like_env_id(env_id_clean) and (not _is_pusht_env_id(env_id_clean)):
        try:
            import gymnasium_robotics as gr
            gym.register_envs(gr)
        except Exception as e:
            raise RuntimeError(
                "Failed to import/register gymnasium_robotics. "
                "Install it (and mujoco) first: `pip install gymnasium-robotics mujoco`.\n"
                f"Original error: {e}"
            )

    make_kwargs: Dict[str, Any] = {}
    tasks_to_keep: Optional[List[str]] = None
    if _is_kitchen_env_id(env_id_clean):
        make_kwargs = _kitchen_make_kwargs_minari_complete_v2()
        tasks_to_keep = list(make_kwargs["tasks_to_complete"])

    # Try to create env (kitchen with kwargs if applicable)
    try:
        if make_kwargs:
            env = gym.make(env_id_clean, **make_kwargs)
        else:
            env = gym.make(env_id_clean)
    except gym.error.NameNotFound as e:
        available = sorted(list(gym.envs.registry.keys()))
        hint = (
            f"\n[Hint] Use a valid id like 'AntMaze_UMaze-v4', 'PointMaze_UMaze-v3', "
            f"or 'FrankaKitchen-v1' (depending on your gymnasium-robotics version).\n"
            f"Some registered ids (first 60):\n  {available[:60]}"
        )
        raise gym.error.NameNotFound(f"{e}{hint}")
    except TypeError:
        # Some gymnasium-robotics versions may not accept all kwargs; fall back progressively.
        if make_kwargs:
            try:
                env = gym.make(env_id_clean, tasks_to_complete=list(make_kwargs["tasks_to_complete"]))
            except TypeError:
                env = gym.make(env_id_clean)
        else:
            env = gym.make(env_id_clean)

    # Kitchen obs adaptation (optional SB3-friendly flatten; also enforces task subset dim)
    env = _maybe_wrap_franka_kitchen_obs(env, tasks_to_keep=tasks_to_keep)

    if seed is not None:
        try:
            _obs, _info = env.reset(seed=seed)
        except TypeError:
            _obs, _info = env.reset()
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(seed)
        if hasattr(env.observation_space, "seed"):
            env.observation_space.seed(seed)
    return env



def evaluate_policy_receding(
    model,
    env: gym.Env,
    episodes: int,
    obs_horizon: int,
    pred_horizon: int,
    action_horizon: int,
    device,
    base_seed: Optional[int] = None,
    metric: str = "auto",   # "auto" | "success" | "return" | "tasks" | "last"
) -> float:
    """
    Receding-horizon rollout with Diffusion Policy.

    Auto metric switching:
      - Kitchen        : "last"    (mean final-step environment reward)
      - PushT          : "last"    (mean final-step overlap/coverage reward)
      - AdroitHandPen  : "return"  (mean cumulative reward per episode)
      - Others         : "success" (maze-style success)

    Notes:
      - In Kitchen and PushT, we use only the reward at the final rollout timestep.
      - For AdroitHandPen, we use episode cumulative reward, then average across episodes.
    """
    assert action_horizon <= pred_horizon
    is_discrete = isinstance(env.action_space, gym.spaces.Discrete)

    env_id = _infer_env_id_from_env(env)
    is_kitchen = _is_kitchen_env_id(env_id)
    is_pusht = _is_pusht_env_id(env_id)
    is_adroit_hand_pen = _is_adroit_hand_pen_env_id(env_id)

    if metric is None:
        metric = "auto"
    metric_l = str(metric).lower()

    if metric_l == "auto":
        if is_kitchen:
            metric_l = "last"
        elif is_pusht:
            metric_l = "last"
        elif is_adroit_hand_pen:
            metric_l = "return"
        else:
            metric_l = "success"

    # Kitchen evaluation is always final-timestep reward for this pipeline.
    if is_kitchen and metric_l in ("success", "return", "tasks", "last_reward", "final", "final_reward"):
        print(f"[Eval] Kitchen detected: overriding metric {metric_l} -> last (final-timestep reward).")
        metric_l = "last"

    # PushT: treat "return" as "last" (you asked for last-step coverage)
    if is_pusht and metric_l in ("return", "last", "last_reward", "final", "final_reward"):
        metric_l = "last"

    # If someone forces "success" on PushT, override to "last"
    if is_pusht and metric_l == "success":
        print("[Eval] PushT detected: overriding metric success -> last.")
        metric_l = "last"

    # AdroitHandPen: always use cumulative reward / mean return
    if is_adroit_hand_pen and metric_l in ("success", "tasks", "last", "last_reward", "final", "final_reward"):
        print("[Eval] AdroitHandPen detected: overriding metric -> return (episode cumulative reward).")
        metric_l = "return"

    low, high = None, None
    if not is_discrete:
        low = torch.tensor(env.action_space.low, dtype=torch.float32)
        high = torch.tensor(env.action_space.high, dtype=torch.float32)

    success_flags: List[float] = []
    returns: List[float] = []
    task_counts: List[float] = []
    last_rewards: List[float] = []  # Kitchen/PushT: store final-step reward per episode

    for ep in range(int(episodes)):
        if base_seed is not None:
            try:
                obs_raw, info = env.reset(seed=int(base_seed) + ep)
            except TypeError:
                obs_raw, info = env.reset()
        else:
            obs_raw, info = env.reset()

        obs = canonicalize_obs(obs_raw)

        # init To-frame buffer
        if isinstance(obs, dict):
            buf = [{k: v.copy() for k, v in obs.items()} for _ in range(obs_horizon)]
        else:
            buf = [obs.copy() for _ in range(obs_horizon)]

        done = False
        ep_success = bool(info.get("success", False)) if isinstance(info, dict) else False
        ep_return = 0.0
        ep_last_reward = 0.0  # last step reward (Kitchen/PushT metric)

        completed_tasks: Set[str] = set()
        saw_task_info = False
        if is_kitchen and metric_l == "tasks":
            completed_tasks, saw0 = _kitchen_update_completed_tasks_from_info(completed_tasks, info)
            saw_task_info = saw_task_info or saw0

        while not done:
            if isinstance(buf[0], dict):
                obs_seq = _build_obs_window_dict(buf, device=device)
                act_seq = model.sample_actions(obs_seq, device=device)[0]  # [Tp, A]
            else:
                obs_seq = _build_obs_window_tensor(buf, device=device)
                act_seq = model.sample_actions(obs_seq, device=device)[0]

            for i in range(action_horizon):
                if is_discrete:
                    a = int(torch.argmax(act_seq[i]).item())
                else:
                    a_t = act_seq[i].detach().cpu()
                    if low is not None and high is not None:
                        a_t = torch.clamp(a_t, low, high)
                    a = a_t.numpy()

                next_obs_raw, rew, terminated, truncated, info = env.step(a)
                done = bool(terminated or truncated)

                r = float(rew)
                ep_return += r
                ep_last_reward = r  # always track last observed reward

                # Kitchen tasks metric: update completed task set from info
                if is_kitchen and metric_l == "tasks":
                    completed_tasks, saw1 = _kitchen_update_completed_tasks_from_info(completed_tasks, info)
                    saw_task_info = saw_task_info or saw1

                # success from info['success'] (preferred)
                if metric_l == "success":
                    if isinstance(info, dict) and "success" in info:
                        ep_success = ep_success or bool(info["success"])
                    else:
                        # fallback dist check (works if goals exist)
                        if isinstance(next_obs_raw, dict) and ("achieved_goal" in next_obs_raw) and ("desired_goal" in next_obs_raw):
                            ag = _concat_goal_vec(next_obs_raw.get("achieved_goal", None))
                            dg = _concat_goal_vec(next_obs_raw.get("desired_goal", None))
                            if ag is not None and dg is not None and ag.shape == dg.shape:
                                dist = np.linalg.norm(ag - dg)
                                ep_success = ep_success or (dist <= 0.45)

                obs = canonicalize_obs(next_obs_raw)

                # slide window
                if isinstance(obs, dict):
                    buf.pop(0)
                    buf.append({k: v.copy() for k, v in obs.items()})
                else:
                    buf.pop(0)
                    buf.append(obs.copy())

                if done:
                    break

        if metric_l == "success":
            success_flags.append(1.0 if ep_success else 0.0)
        elif metric_l == "tasks":
            if saw_task_info:
                task_counts.append(float(len(completed_tasks)))
            else:
                # Fallback: if task info missing, use return as proxy
                task_counts.append(float(ep_return))
        elif metric_l == "last":
            last_rewards.append(float(ep_last_reward))
        else:
            # "return": mean episode cumulative reward
            returns.append(float(ep_return))

    if metric_l == "success":
        succ = float(np.mean(success_flags)) if success_flags else 0.0
        print(f"[Eval] metric=success_rate, success_rate={succ:.3f}")
        return succ

    if metric_l == "tasks":
        mean_tasks = float(np.mean(task_counts)) if task_counts else 0.0
        print(f"[Eval] metric=mean_tasks_completed, mean_tasks_completed={mean_tasks:.3f}")
        return mean_tasks

    if metric_l == "last":
        mean_last = float(np.mean(last_rewards)) if last_rewards else 0.0
        print(f"[Eval] metric=mean_final_reward, mean_final_reward={mean_last:.3f}")
        return mean_last

    mean_ret = float(np.mean(returns)) if returns else 0.0
    print(f"[Eval] metric=mean_cumulative_reward, mean_cumulative_reward={mean_ret:.3f}")
    return mean_ret



def _prepare_obs_for_collate(x: torch.Tensor) -> torch.Tensor:
    """Remove leading batch dimension of size 1 so stacking produces [B, ...] not [B,1,...]."""
    if x.dim() >= 1 and x.size(0) == 1:
        return x.squeeze(0)
    return x


def _collate_obs_list(obs_list):
    """
    Collate a list of obs_t (dict or tensor) into a batched obs.
    Supports nested dict recursively.
    """
    if len(obs_list) == 0:
        return None

    o0 = obs_list[0]
    if isinstance(o0, dict):
        out = {}
        for k in o0.keys():
            vs = [o[k] for o in obs_list]
            if isinstance(vs[0], dict):
                out[k] = _collate_obs_list(vs)
            else:
                ts = [_prepare_obs_for_collate(v) for v in vs]
                out[k] = torch.stack(ts, dim=0)
        return out
    else:
        vs = [_prepare_obs_for_collate(o) for o in obs_list]
        return torch.stack(vs, dim=0)
