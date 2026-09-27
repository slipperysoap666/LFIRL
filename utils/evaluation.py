from __future__ import annotations
from typing import Any, Dict, Optional, Union, Callable, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList


def _to_device(device: Union[str, torch.device]) -> torch.device:
    """
    Normalize a device spec to torch.device.
    NOTE: one-argument signature (this fixes your TypeError).
    """
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _is_dict_space(space: gym.Space) -> bool:
    return isinstance(space, gym.spaces.Dict)


def _policy_for_space(space: gym.Space) -> str:
    """
    Choose SB3 policy depending on obs space.
    - Dict obs -> 'MultiInputPolicy'
    - Array obs -> 'MlpPolicy'
    """
    if _is_dict_space(space):
        return "MultiInputPolicy"
    return "MlpPolicy"


def _is_kitchen_env_id(env_id: str) -> bool:
    s = str(env_id).lower()
    return ("kitchen" in s) or ("frankakitchen" in s) or ("franka_kitchen" in s)


def _is_pusht_env_id(env_id: str) -> bool:
    """
    Detect PushT env ids, e.g. "gym_pusht/PushT-v0".
    """
    s = str(env_id).lower()
    return ("pusht" in s) or ("gym_pusht" in s)


def _is_adroit_hand_pen_env_id(env_id: str) -> bool:
    """
    Detect Adroit Hand Pen env ids, e.g.:
      - AdroitHandPen-v1
      - AdroitHandPenSparse-v1
      - adroit/hand_pen
    """
    s = str(env_id).lower()
    return (
        ("adroit" in s and "pen" in s)
        or ("handpen" in s)
        or ("hand_pen" in s)
    )


def _to_dense_maze_env_id(env_id: str) -> str:
    """
    If the env is a PointMaze sparse-reward id like "PointMaze_Medium-v3",
    switch it to the corresponding dense-reward id "PointMaze_MediumDense-v3".
    Leave all other env ids unchanged.
    """
    s = str(env_id)
    s_l = s.lower()

    if ("pointmaze_" not in s_l) or ("dense" in s_l):
        return s

    if "-" not in s:
        return s

    base, suffix = s.rsplit("-", 1)
    return f"{base}Dense-{suffix}"


def _metric_name_for_env(env_id: str, metric_l: str) -> str:
    """
    Pretty metric name for logging.
    """
    if metric_l == "tasks":
        return "mean_tasks_completed"
    if metric_l == "last":
        if _is_kitchen_env_id(env_id):
            return "mean_final_reward"
        return "mean_last_overlap"
    if metric_l == "return":
        if _is_adroit_hand_pen_env_id(env_id):
            return "mean_cumulative_reward"
        return "mean_return"
    return "success_rate"


def _first_linear_in_features(sequential: nn.Sequential) -> Optional[int]:
    """Return in_features of the first nn.Linear in a Sequential."""
    for m in sequential:
        if isinstance(m, nn.Linear):
            return m.in_features
    return None


def _infer_time_repeats_for_flat(reward_net, S: int) -> int:
    """
    If the reward net's flat-obs MLP was built on (To*S), infer To to tile obs to [1, To, S].
    Otherwise return 1.
    """
    enc = getattr(reward_net, "obs_encoder", None)
    if enc is None or getattr(enc, "_flat_mlp", None) is None:
        return 1
    inf = _first_linear_in_features(enc._flat_mlp.net)  # type: ignore[attr-defined]
    if inf is None or inf <= S:
        return 1
    if inf % S == 0:
        return int(inf // S)
    return 1


def _infer_time_repeats_for_key(reward_net, key: str, Sk: int) -> int:
    """
    If the reward net's per-key MLP for `key` was built on (To*Sk), infer To.
    """
    enc = getattr(reward_net, "obs_encoder", None)
    if enc is None:
        return 1
    per_mlps = getattr(enc, "_per_key_mlps", None)
    if per_mlps is None or key not in per_mlps:
        return 1
    mlp = per_mlps[key]
    inf = _first_linear_in_features(mlp.net)
    if inf is None or inf <= Sk:
        return 1
    if inf % Sk == 0:
        return int(inf // Sk)
    return 1


def _to_torch_action(action: np.ndarray, device: torch.device) -> torch.Tensor:
    if isinstance(action, np.ndarray):
        return torch.as_tensor(action, dtype=torch.float32, device=device).view(1, -1)
    return action.to(device).view(1, -1)


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


class RewardNetWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, reward_net: nn.Module, device: Union[str, torch.device] = "cpu"):
        super().__init__(env)
        self.reward_net = reward_net.eval()
        self.device = _to_device(device)
        self.reward_net.to(self.device)

        self._flat_time_repeats_cache: Optional[int] = None
        self._dict_time_repeats_cache: Dict[str, int] = {}
        self._last_obs = None

    def _obs_to_model(self, obs: Union[np.ndarray, Dict[str, np.ndarray]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if isinstance(obs, dict):
            out: Dict[str, torch.Tensor] = {}
            for k, v in obs.items():
                vt = torch.as_tensor(v, dtype=torch.float32, device=self.device)
                if vt.dim() == 1:
                    vt_b = vt.view(1, -1)
                elif vt.dim() == 2 and vt.size(0) != 1:
                    vt_b = vt
                else:
                    vt_b = vt

                Sk = vt_b.shape[-1]
                To_k = self._dict_time_repeats_cache.get(k)
                if To_k is None:
                    To_k = _infer_time_repeats_for_key(self.reward_net, k, Sk)
                    self._dict_time_repeats_cache[k] = To_k

                if To_k > 1 and vt_b.dim() == 2 and vt_b.size(0) == 1:
                    vt_b = vt_b.unsqueeze(1).repeat(1, To_k, 1)

                out[k] = vt_b
            return out

        vt = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if vt.dim() == 1:
            vt_b = vt.view(1, -1)
        else:
            vt_b = vt

        if self._flat_time_repeats_cache is None:
            self._flat_time_repeats_cache = _infer_time_repeats_for_flat(self.reward_net, vt_b.shape[-1])

        To = self._flat_time_repeats_cache
        if To is not None and To > 1 and vt_b.dim() == 2:
            vt_b = vt_b.unsqueeze(1).repeat(1, To, 1)
        return vt_b

    def step(self, action):
        prev_obs = self._last_obs
        next_obs, _, terminated, truncated, info = self.env.step(action)

        # If step() is called before reset(), fall back defensively.
        if prev_obs is None:
            prev_obs = next_obs

        obs_t = self._obs_to_model(prev_obs)
        a_t = _to_torch_action(action, self.device)

        with torch.no_grad():
            r_t = self.reward_net(obs_t, a_t).view(-1).item()

        self._last_obs = next_obs
        return next_obs, float(r_t), terminated, truncated, info

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}
        self._last_obs = obs
        return obs, info


class TrainingProgressCallback(BaseCallback):
    """
    Callback that prints training progress in percent.
    It prints at most once per 1% of total timesteps to avoid spamming.
    """

    def __init__(self, total_timesteps: int, verbose: int = 1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.last_progress = -1

    def _on_step(self) -> bool:
        if self.total_timesteps <= 0:
            return True

        progress = int(100 * self.num_timesteps / self.total_timesteps)
        if progress > 100:
            progress = 100

        if progress != self.last_progress:
            self.last_progress = progress
            if self.verbose > 0:
                print(f"[Train] progress: {progress:3d}% ({self.num_timesteps}/{self.total_timesteps} timesteps)")
        return True


class PeriodicEvalCallback(BaseCallback):
    """
    Periodically evaluate the current policy on the ORIGINAL environment.

    Auto metric switching:
      - Kitchen        : mean_final_reward
      - PushT          : mean_last_overlap
      - AdroitHandPen  : mean_cumulative_reward
      - Others         : success_rate
    """

    def __init__(
        self,
        env_id: str,
        make_env_fn: Callable[[str], gym.Env],
        eval_episodes: int = 10,
        eval_every: int = 50_000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.env_id = env_id
        self.make_env_fn = make_env_fn
        self.eval_episodes = int(eval_episodes)
        self.eval_every = int(eval_every)
        self._last_eval_at = 0

    def _on_step(self) -> bool:
        if self.eval_every <= 0:
            return True

        if (self.num_timesteps - self._last_eval_at) >= self.eval_every:
            self._last_eval_at = self.num_timesteps

            metric_val = evaluate_on_original_env(
                self.model,
                self.env_id,
                self.make_env_fn,
                eval_episodes=self.eval_episodes,
                deterministic=False,
                metric="auto",
            )

            if _is_kitchen_env_id(self.env_id):
                metric_name = "mean_final_reward"
            elif _is_pusht_env_id(self.env_id):
                metric_name = "mean_last_overlap"
            elif _is_adroit_hand_pen_env_id(self.env_id):
                metric_name = "mean_cumulative_reward"
            else:
                metric_name = "success_rate"

            if self.verbose > 0:
                print(f"[SAC-PeriodicEval] timesteps={self.num_timesteps}, {metric_name}={metric_val:.3f}")

        return True


def _make_wrapped_env(
    env_id: str,
    make_env_fn: Callable[[str], gym.Env],
    reward_net: nn.Module,
    device: Union[str, torch.device],
) -> gym.Env:
    base_env = make_env_fn(_to_dense_maze_env_id(env_id))
    wrapped = RewardNetWrapper(base_env, reward_net=reward_net, device=device)
    return Monitor(wrapped)


def train_sac_on_wrapped_env(
    env: gym.Env,
    seed: int = 0,
    total_timesteps: int = 400_000,
    sb3_kwargs: Optional[Dict[str, Any]] = None,
    extra_callback: Optional[BaseCallback] = None,
) -> SAC:
    """
    Train a SAC agent on the reward-wrapped environment.
    Automatically chooses MultiInputPolicy for dict obs.
    """
    sb3_kwargs = sb3_kwargs or {}
    policy = _policy_for_space(env.observation_space)

    vec_env = DummyVecEnv([lambda: env])

    default_sac_args: Dict[str, Any] = dict(
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=10_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        target_update_interval=1,
        verbose=1,
        seed=seed,
        device="auto",
    )
    default_sac_args.update(sb3_kwargs)

    model = SAC(policy, vec_env, **default_sac_args)

    progress_callback = TrainingProgressCallback(total_timesteps=total_timesteps, verbose=1)

    if extra_callback is not None:
        callback: BaseCallback = CallbackList([progress_callback, extra_callback])
    else:
        callback = progress_callback

    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callback)
    return model


def evaluate_on_original_env(
    model: SAC,
    env_id: str,
    make_env_fn: Callable[[str], gym.Env],
    eval_episodes: int = 10,
    deterministic: bool = False,
    metric: str = "auto",
) -> float:
    """
    Evaluate a trained policy on the ORIGINAL environment.

    Auto metric switching:
      - Kitchen        : mean final-timestep reward
      - PushT          : mean_last_overlap (reward at final timestep)
      - AdroitHandPen  : mean cumulative reward per episode
      - Others         : success_rate

    Returns:
        float: success_rate / mean_tasks_completed / mean_last_overlap /
               mean_cumulative_reward depending on metric/env.
    """
    metric_l = str(metric).lower() if metric is not None else "auto"
    is_kitchen = _is_kitchen_env_id(env_id)
    is_pusht = _is_pusht_env_id(env_id)
    is_adroit_hand_pen = _is_adroit_hand_pen_env_id(env_id)

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

    # Enforce: PushT uses final-timestep overlap (reward) by default.
    # Also map "return"/"success" to "last" for PushT to avoid wrong metrics.
    if is_pusht and metric_l in ("success", "return"):
        print(f"[Eval] PushT detected: overriding metric {metric_l} -> last (final-timestep overlap).")
        metric_l = "last"

    # Enforce: AdroitHandPen uses mean cumulative reward.
    if is_adroit_hand_pen and metric_l in ("success", "tasks", "last", "last_reward", "final", "final_reward"):
        print(f"[Eval] AdroitHandPen detected: overriding metric {metric_l} -> return (episode cumulative reward).")
        metric_l = "return"

    env = make_env_fn(env_id)

    is_discrete = isinstance(env.action_space, gym.spaces.Discrete)

    low, high = (None, None)
    if not is_discrete:
        low = np.asarray(env.action_space.low, dtype=np.float32)
        high = np.asarray(env.action_space.high, dtype=np.float32)

    success_key = "success"
    goal_dist_threshold = 0.5

    success_flags: list[float] = []
    returns: list[float] = []       # used only for metric_l == "return"
    task_counts: list[float] = []   # used only for metric_l == "tasks" (kitchen)
    last_overlaps: list[float] = [] # used only for metric_l == "last" (kitchen/pusht)

    for _ in range(int(eval_episodes)):
        try:
            obs, info = env.reset()
        except TypeError:
            obs = env.reset()
            info = {}

        done = False
        ep_success = bool(info.get(success_key, False)) if isinstance(info, dict) else False
        ep_return = 0.0
        ep_last_reward = 0.0  # last-timestep reward for Kitchen/PushT

        completed_tasks: Set[str] = set()
        saw_task_info = False
        if is_kitchen and metric_l == "tasks":
            completed_tasks, saw0 = _kitchen_update_completed_tasks_from_info(completed_tasks, info)
            saw_task_info = saw_task_info or saw0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)

            if is_discrete:
                action = int(action)
            else:
                action = np.asarray(action, dtype=np.float32)
                if low is not None and high is not None:
                    action = np.clip(action, low, high)

            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = bool(terminated or truncated)

            r = float(reward)
            ep_return += r
            ep_last_reward = r

            if is_kitchen and metric_l == "tasks":
                completed_tasks, saw1 = _kitchen_update_completed_tasks_from_info(completed_tasks, step_info)
                saw_task_info = saw_task_info or saw1

            if metric_l == "success":
                if isinstance(step_info, dict) and (success_key in step_info):
                    ep_success = ep_success or bool(step_info[success_key])
                else:
                    check_obs = next_obs
                    if isinstance(check_obs, dict) and ("achieved_goal" in check_obs) and ("desired_goal" in check_obs):
                        try:
                            dist = np.linalg.norm(
                                np.asarray(check_obs["achieved_goal"]) - np.asarray(check_obs["desired_goal"])
                            )
                            if np.isfinite(dist):
                                ep_success = ep_success or (float(dist) <= goal_dist_threshold)
                        except Exception:
                            pass

            obs = next_obs

        if metric_l == "success":
            success_flags.append(1.0 if ep_success else 0.0)
        elif metric_l == "tasks":
            if saw_task_info:
                task_counts.append(float(len(completed_tasks)))
            else:
                task_counts.append(float(ep_return))
        elif metric_l == "last":
            last_overlaps.append(float(ep_last_reward))
        else:
            returns.append(float(ep_return))

    if metric_l == "success":
        succ_rate = float(np.mean(success_flags)) if success_flags else 0.0
        print(f"[Eval] metric=success_rate, success_rate={succ_rate:.3f}")
        return succ_rate

    if metric_l == "tasks":
        mean_tasks = float(np.mean(task_counts)) if task_counts else 0.0
        print(f"[Eval] metric=mean_tasks_completed, mean_tasks_completed={mean_tasks:.3f}")
        return mean_tasks

    if metric_l == "last":
        mean_last = float(np.mean(last_overlaps)) if last_overlaps else 0.0
        if is_kitchen:
            print(f"[Eval] metric=mean_final_reward, mean_final_reward={mean_last:.3f}")
        else:
            print(f"[Eval] metric=mean_last_overlap, mean_last_overlap={mean_last:.3f}")
        return mean_last

    mean_ret = float(np.mean(returns)) if returns else 0.0
    if is_adroit_hand_pen:
        print(f"[Eval] metric=mean_cumulative_reward, mean_cumulative_reward={mean_ret:.3f}")
    else:
        print(f"[Eval] metric=mean_return, mean_return={mean_ret:.3f}")
    return mean_ret


def evaluate_reward_via_sac(
    *,
    env_id: str,
    make_env_fn: Callable[[str], gym.Env],
    reward_net: nn.Module,
    total_timesteps: int = 400_000,
    eval_episodes: int = 10,
    seed: int = 0,
    device: Union[str, torch.device] = "cpu",
    sb3_kwargs: Optional[Dict[str, Any]] = None,
) -> float:
    """
    1) Wrap env to use r_theta(s,a) from reward_net as reward.
    2) Train SAC on the wrapped env.
    3) Periodically evaluate the current policy on the ORIGINAL env.
    4) After training, evaluate once more and return the final metric.

    Metric (auto):
      - Kitchen        : mean final-timestep reward
      - PushT          : mean_last_overlap (reward at final timestep)
      - AdroitHandPen  : mean_cumulative_reward
      - Others         : success_rate
    """
    device = _to_device(device)

    wrapped_env = _make_wrapped_env(env_id, make_env_fn, reward_net, device)

    periodic_eval_cb = PeriodicEvalCallback(
        env_id=env_id,
        make_env_fn=make_env_fn,
        eval_episodes=eval_episodes,
        eval_every=50_000,
        verbose=1,
    )

    model = train_sac_on_wrapped_env(
        wrapped_env,
        seed=seed,
        total_timesteps=total_timesteps,
        sb3_kwargs=sb3_kwargs,
        extra_callback=periodic_eval_cb,
    )

    final_val = evaluate_on_original_env(
        model,
        env_id,
        make_env_fn,
        eval_episodes=eval_episodes,
        deterministic=False,
        metric="auto",
    )
    return final_val
