import os
import re
import zipfile
import urllib.request
from typing import Optional, List, Tuple, Dict, Any, Mapping

import numpy as np
import gymnasium as gym
import torch

from utils.utils import KITCHEN_MINARI_COMPLETE_V2_TASKS, make_env


FLAT_KEY_SEP = "__"

PUSHT_ZIP_URL = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
PUSHT_DEFAULT_ZARR_REL = os.path.join("data", "pusht", "pusht_cchi_v7_replay.zarr")


def _project_root() -> str:
    """
    datasets/dataset.py -> project root is one level up (../).
    Adjust if your repo layout differs.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _is_pusht_env(env_id: str) -> bool:
    name = (env_id or "").lower()
    return "pusht" in name


def _is_kitchen_env(env_id: str) -> bool:
    name = (env_id or "").lower()
    return ("kitchen" in name) or ("frankakitchen" in name) or ("franka_kitchen" in name)


def _safe_id_for_filename(s: str) -> str:
    """
    Convert an env_id (which may contain '/', ':', etc.) into a filesystem-safe filename stem.
    - Windows forbids: < > : " / \\ | ? *
    - Also avoid spaces.
    """
    s = str(s)
    # Replace any forbidden/reserved path characters with a double-underscore
    s = re.sub(r'[<>:"/\\|?*]+', "__", s)
    # Normalize whitespace
    s = re.sub(r"\s+", "_", s).strip("_")
    # Avoid empty
    return s if s else "env"


def _map_env_to_minari_id(env_id: str) -> Optional[str]:
    """
    Best-effort mapping from env_id to Minari dataset id.
    Extend as needed.
    """
    name = env_id.lower()

    if re.search(r"antmaze[_\-]umaze\-v\d+", name):
        return "D4RL/antmaze/umaze-v1"
    if re.search(r"antmaze[_\-]medium\-v\d+", name):
        return "D4RL/antmaze/medium-play-v1"
    if re.search(r"antmaze[_\-]large\-v\d+", name):
        return "D4RL/antmaze/large-play-v1"

    if "pointmaze" in name:
        if "umaze" in name:
            return "D4RL/pointmaze/umaze-v2"
        if "medium" in name:
            return "D4RL/pointmaze/medium-v2"
        if "large" in name:
            return "D4RL/pointmaze/large-v2"

    if "kitchen" in name:
        return "D4RL/kitchen/complete-v2"

    if "pen" in name:
        return "D4RL/pen/expert-v2"

    return None


class OfflineDataset:
    """
    Load the FULL dataset once into expert_data/{safe_env_id}_trajectories.npz.
    Then, at initialization, choose num_expert episodes and physically shrink.

    Extended:
      - If env_id is PushT (e.g., "gym_pusht/PushT-v0") and no Minari dataset exists,
        download Diffusion-Policy PushT demos (pusht.zip), extract, convert from zarr,
        and save into your standard trajectories.npz format.
    """

    def __init__(
        self,
        env_id: str,
        env: Optional[gym.Env] = None,
        num_expert: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.env_id = env_id

        # Put expert_data under project root (more robust than CWD)
        self.expert_dir = os.path.join(_project_root(), "expert_data")
        os.makedirs(self.expert_dir, exist_ok=True)

        self.num_expert = int(num_expert) if (num_expert is not None and num_expert > 0) else None
        self.subset_seed = int(seed) if seed is not None else 23

        # IMPORTANT: env_id may contain '/', which becomes a subdirectory on Windows if not sanitized.
        self.env_tag = _safe_id_for_filename(env_id)
        self.traj_path = os.path.join(self.expert_dir, f"{self.env_tag}_trajectories.npz")

        self.env = env if env is not None else make_env(self.env_id)

        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.is_discrete = True
            self.action_dim = self.env.action_space.n
        else:
            assert isinstance(self.env.action_space, gym.spaces.Box), \
                "OfflineDataset: unsupported action space type."
            self.is_discrete = False
            self.action_dim = int(np.prod(self.env.action_space.shape))

        # If file not materialized, build it now
        if not os.path.exists(self.traj_path):
            if _is_pusht_env(self.env_id):
                self._materialize_from_pusht()
            else:
                self._materialize_from_minari()

        data = np.load(self.traj_path, allow_pickle=True)
        self.actions = data["actions"]          # [N, A]
        self.rewards = data["rewards"]          # [N]
        self.dones = data["dones"]              # [N]
        self.next_states = data["next_states"]  # [N, S]
        self.states = data["states"]            # [N, S]

        # optional dict-obs (flattened leaf keys)
        self.obs_keys: Optional[List[str]] = None
        self.states_dict: Optional[Dict[str, np.ndarray]] = None
        self.obs_slices: Optional[Dict[str, slice]] = None

        if "obs_keys" in data.files:
            self.obs_keys = list(data["obs_keys"].tolist())
            states_dict = {}
            for k in self.obs_keys:
                name = f"states__{k}"
                if name in data.files:
                    states_dict[k] = data[name]
            self.states_dict = states_dict if len(states_dict) > 0 else None

            if "obs_shapes" in data.files:
                obs_shapes = list(data["obs_shapes"].tolist())
                offsets, total = [], 0
                for shape in obs_shapes:
                    Sk = int(np.prod(shape))
                    offsets.append((total, total + Sk))
                    total += Sk
                self.obs_slices = {k: slice(lo, hi) for k, (lo, hi) in zip(self.obs_keys, offsets)}

        self.episode_ranges: List[Tuple[int, int]] = self._compute_episode_ranges(self.dones)  # inclusive [s, e]
        chosen = self._choose_episode_ranges(self.episode_ranges, self.num_expert, seed=self.subset_seed)
        self._shrink_to_selected(chosen)

        print(f"[Dataset] Traj file: {self.traj_path}")
        print(f"[Dataset] Final: episodes={len(self.episode_ranges)}  steps={self.states.shape[0]}")

    @staticmethod
    def _compute_episode_ranges(dones: np.ndarray) -> List[Tuple[int, int]]:
        """
        From a boolean 'dones' over the flat buffer, build inclusive episode ranges (start, end).
        'True' marks the last step of an episode. If tail has no True, close it at N-1.
        """
        ranges: List[Tuple[int, int]] = []
        start = 0
        N = int(dones.shape[0])
        for i in range(N):
            if bool(dones[i]):
                ranges.append((start, i))
                start = i + 1
        if start < N:
            ranges.append((start, N - 1))
        return ranges

    @staticmethod
    def _gather_indices_for_episodes(ranges: List[Tuple[int, int]]) -> np.ndarray:
        """Concatenate all timestep indices of the provided inclusive episode ranges."""
        if not ranges:
            return np.zeros((0,), dtype=np.int64)
        parts = [np.arange(s, e + 1, dtype=np.int64) for (s, e) in ranges]
        return np.concatenate(parts, axis=0)

    def _choose_episode_ranges(
        self,
        ranges: List[Tuple[int, int]],
        num_expert: Optional[int],
        seed: int = 23,
    ) -> List[Tuple[int, int]]:
        """Randomly choose `num_expert` episodes from full `ranges` (inclusive ends)."""
        total_eps = len(ranges)
        if total_eps == 0:
            print("[Dataset] Warning: no episodes found.")
            return []
        if (num_expert is None) or (num_expert >= total_eps):
            print(f"[Dataset] Episodes available: {total_eps} | Using: {total_eps} (all)")
            return list(ranges)
        k = int(num_expert)
        rng = np.random.default_rng(seed)
        idx = rng.choice(total_eps, size=k, replace=False)
        chosen = [ranges[i] for i in idx]
        print(f"[Dataset] Episodes available: {total_eps} | Using: {k}")
        return chosen

    def _shrink_to_selected(self, chosen_ranges: List[Tuple[int, int]]) -> None:
        """
        Physically subset all arrays to only chosen episodes (keep original order).
        Recompute episode_ranges/train_indices/num_samples accordingly.
        NOTE: ranges use inclusive ends (start, end).
        """
        if not chosen_ranges:
            self.actions = self.actions[:0]
            self.rewards = self.rewards[:0]
            self.dones = self.dones[:0]
            self.next_states = self.next_states[:0]
            self.states = self.states[:0]
            if self.states_dict is not None:
                for k in list(self.states_dict.keys()):
                    self.states_dict[k] = self.states_dict[k][:0]
            self.episode_ranges = []
            self.train_indices = np.zeros((0,), dtype=np.int64)
            self.num_samples = 0
            return

        idx = self._gather_indices_for_episodes(chosen_ranges)

        self.actions = self.actions[idx]
        self.rewards = self.rewards[idx]
        self.dones = self.dones[idx]
        self.next_states = self.next_states[idx]
        self.states = self.states[idx]
        if self.states_dict is not None:
            for k in list(self.states_dict.keys()):
                self.states_dict[k] = self.states_dict[k][idx]

        self.episode_ranges = self._compute_episode_ranges(self.dones)
        self.train_indices = np.arange(self.states.shape[0], dtype=np.int64)
        self.num_samples = int(self.train_indices.size)

    @staticmethod
    def _flatten_nested_dict_leaves(
        d: Mapping[str, Any],
        prefix: str = "",
        sep: str = FLAT_KEY_SEP,
    ) -> Dict[str, np.ndarray]:
        """
        Flatten a (possibly nested) dict into leaf arrays, using sep to join keys.
        Leaf values must be array-like (np.ndarray / list / scalar).
        """
        out: Dict[str, np.ndarray] = {}
        for k, v in d.items():
            kk = f"{prefix}{sep}{k}" if prefix else str(k)
            if isinstance(v, dict):
                out.update(OfflineDataset._flatten_nested_dict_leaves(v, prefix=kk, sep=sep))
            else:
                out[kk] = np.asarray(v)
        return out

    @staticmethod
    def _flatten_obs_to_vec(
        obs_like: Any,
        key_order: Optional[Tuple[str, ...]] = None,
        sep: str = FLAT_KEY_SEP,
    ) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]], Optional[Tuple[str, ...]]]:
        """
        Convert one step observation into:
          - flat vector (float32, 1D)
          - leaf dict (key -> 1D float32 array), or None if not dict obs
          - key_order used (deterministic, sorted on first use)
        """
        if isinstance(obs_like, dict):
            leaf = OfflineDataset._flatten_nested_dict_leaves(obs_like, sep=sep)
            if key_order is None:
                key_order = tuple(sorted(leaf.keys()))
            else:
                if set(leaf.keys()) != set(key_order):
                    raise RuntimeError(
                        f"Observation dict leaf keys changed across time. "
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

    @staticmethod
    def _concat_selected_goal_tasks(goal_like: Any, tasks: List[str]) -> np.ndarray:
        """
        Kitchen Minari observations expose achieved_goal/desired_goal as nested dicts.
        The runtime env wrapper vectorizes those goals, so materialization must do the
        same thing or pretrained policy keys will not match evaluation-time obs keys.
        """
        if not isinstance(goal_like, dict):
            return np.asarray(goal_like, dtype=np.float32).reshape(-1)

        parts: List[np.ndarray] = []
        for task in tasks:
            if task not in goal_like:
                raise KeyError(
                    f"Kitchen goal dict missing task '{task}'. "
                    f"Available tasks={sorted(goal_like.keys())}"
                )
            parts.append(np.asarray(goal_like[task], dtype=np.float32).reshape(-1))
        if len(parts) == 0:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)

    @staticmethod
    def _canonicalize_kitchen_obs(obs_like: Any) -> Any:
        """
        Keep Franka Kitchen data aligned to make_env(...):
          - only kettle/light switch/microwave/slide cabinet goals
          - achieved_goal and desired_goal are vector leaves
          - observation stays as the proprioceptive/state vector
        """
        if not isinstance(obs_like, dict):
            return obs_like

        out: Dict[str, Any] = {}
        for k, v in obs_like.items():
            if k in ("achieved_goal", "desired_goal"):
                out[k] = OfflineDataset._concat_selected_goal_tasks(
                    v,
                    list(KITCHEN_MINARI_COMPLETE_V2_TASKS),
                )
            elif isinstance(v, dict):
                out[k] = OfflineDataset._flatten_obs_to_vec(v)[0]
            else:
                out[k] = np.asarray(v, dtype=np.float32)
        return out

    @staticmethod
    def _slice_step(obs_buf, t_idx: int):
        """
        Slice a time-major observation container at timestep t_idx.

        Supports:
          - array obs: obs_buf is np.ndarray/list with shape [T+1, ...]
          - dict obs: obs_buf is dict(key -> array [T+1, ...]) or nested dict of arrays
        """
        if isinstance(obs_buf, dict):
            out = {}
            for k, v in obs_buf.items():
                if isinstance(v, dict):
                    out[k] = OfflineDataset._slice_step(v, t_idx)
                else:
                    v_arr = np.asarray(v)
                    out[k] = v_arr[t_idx]
            return out
        else:
            arr = np.asarray(obs_buf)
            return arr[t_idx]

    # PushT materialization (Diffusion Policy data: pusht.zip -> zarr -> npz)
    def _download_if_needed(self, url: str, dst_path: str) -> None:
        os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
        if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
            return

        print(f"[PushT] Downloading: {url}")
        print(f"[PushT]   -> {dst_path}")

        def _reporthook(block_num: int, block_size: int, total_size: int):
            if total_size <= 0:
                return
            downloaded = block_num * block_size
            pct = min(100.0, 100.0 * downloaded / total_size)
            if block_num % 50 == 0:
                print(f"[PushT] Download progress: {pct:.1f}%")

        urllib.request.urlretrieve(url, dst_path, reporthook=_reporthook)
        print("[PushT] Download done.")

    def _find_zarr_dir(self, base_dir: str) -> Optional[str]:
        """
        Find a *.zarr directory under base_dir.
        Prefer the default expected path if it exists.
        """
        proj = _project_root()
        default_path = os.path.join(proj, PUSHT_DEFAULT_ZARR_REL)
        if os.path.isdir(default_path):
            return default_path

        for root_dir, dirnames, _filenames in os.walk(base_dir):
            for d in dirnames:
                if d.endswith(".zarr"):
                    cand = os.path.join(root_dir, d)
                    if os.path.isdir(os.path.join(cand, "meta")) and os.path.isdir(os.path.join(cand, "data")):
                        return cand
        return None

    def _materialize_from_pusht(self) -> None:
        """
        Download pusht.zip to project root, extract into project_root/data/,
        locate zarr, and convert to expert_data/{safe_env_id}_trajectories.npz.
        """
        try:
            import zarr  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "PushT conversion requires 'zarr'. Install it with: pip install zarr\n"
                f"Import error: {e}"
            )

        proj = _project_root()
        zip_path = os.path.join(proj, "pusht.zip")

        self._download_if_needed(PUSHT_ZIP_URL, zip_path)

        data_dir = os.path.join(proj, "data")
        os.makedirs(data_dir, exist_ok=True)

        # Avoid re-extract every time: check for existing zarr first
        zarr_dir = self._find_zarr_dir(data_dir)
        if zarr_dir is None:
            print(f"[PushT] Extracting {zip_path} -> {data_dir}")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(data_dir)
            print("[PushT] Extract done.")
            zarr_dir = self._find_zarr_dir(data_dir)

        if zarr_dir is None:
            raise FileNotFoundError(
                f"[PushT] Could not locate any .zarr directory under {data_dir} after extraction."
            )

        print(f"[PushT] Using zarr dataset: {zarr_dir}")

        root = zarr.open_group(zarr_dir, mode="r")

        if "data" not in root or "meta" not in root:
            raise RuntimeError(f"[PushT] Invalid zarr structure: missing 'data'/'meta' groups in {zarr_dir}")

        data_g = root["data"]
        meta_g = root["meta"]

        if "action" not in data_g:
            raise RuntimeError("[PushT] zarr missing data/action")
        actions = np.asarray(data_g["action"][:], dtype=np.float32)

        if "state" not in data_g:
            raise RuntimeError("[PushT] zarr missing data/state")
        states_raw = np.asarray(data_g["state"][:], dtype=np.float32)

        if "episode_ends" not in meta_g:
            raise RuntimeError("[PushT] zarr missing meta/episode_ends")
        episode_ends = np.asarray(meta_g["episode_ends"][:], dtype=np.int64).reshape(-1)

        keypoints = None
        if "keypoint" in data_g:
            keypoints = np.asarray(data_g["keypoint"][:], dtype=np.float32)

        rewards = None
        if "reward" in data_g:
            rewards = np.asarray(data_g["reward"][:], dtype=np.float32).reshape(-1)

        N_a = int(actions.shape[0])
        N_s = int(states_raw.shape[0])

        if N_s == N_a + 1:
            states = states_raw[:-1]
            next_states = states_raw[1:]
            N = N_a
        else:
            N = min(N_a, N_s)
            if N != N_a or N != N_s:
                print(f"[PushT] Warning: length mismatch: actions={N_a}, states={N_s}. Truncating to {N}.")
            states = states_raw[:N]
            next_states = np.empty_like(states)

        actions = actions[:N]

        dones = np.zeros((N,), dtype=np.bool_)

        max_end = int(episode_ends.max()) if episode_ends.size > 0 else -1
        if max_end <= 0:
            raise RuntimeError("[PushT] episode_ends looks empty/invalid.")

        if max_end == N:
            ends_exclusive = episode_ends
            end_indices = ends_exclusive - 1
        elif max_end == N - 1:
            end_indices = episode_ends
            ends_exclusive = episode_ends + 1
        else:
            print(f"[PushT] Warning: episode_ends max={max_end} not in {{N, N-1}} (N={N}). Clamping.")
            ends_exclusive = np.clip(episode_ends, 1, N)
            end_indices = ends_exclusive - 1

        end_indices = np.clip(end_indices, 0, N - 1)
        dones[end_indices] = True

        if N_s != N_a + 1:
            start = 0
            for e in ends_exclusive:
                e = int(e)
                e = max(start + 1, min(e, N))
                if e - start >= 2:
                    next_states[start:e - 1] = states[start + 1:e]
                next_states[e - 1] = states[e - 1]
                start = e
            if start < N:
                if N - start >= 2:
                    next_states[start:N - 1] = states[start + 1:N]
                next_states[N - 1] = states[N - 1]

        if rewards is None:
            rewards = np.zeros((N,), dtype=np.float32)
        else:
            rewards = rewards[:N].astype(np.float32)

        save_dict: Dict[str, Any] = dict(
            states=states.astype(np.float32),
            actions=actions.astype(np.float32),
            rewards=rewards.astype(np.float32),
            next_states=next_states.astype(np.float32),
            dones=dones.astype(np.bool_),
        )

        obs_space = getattr(self.env, "observation_space", None)
        if isinstance(obs_space, gym.spaces.Dict):
            need_env_state = "environment_state" in obs_space.spaces
            need_agent_pos = "agent_pos" in obs_space.spaces
            if need_env_state and need_agent_pos:
                if keypoints is None:
                    raise RuntimeError(
                        "[PushT] Env observation_space requires 'environment_state' but zarr has no data/keypoint."
                    )
                agent_pos = states[:, 0:2].astype(np.float32)
                env_state = keypoints[:N].reshape(N, -1).astype(np.float32)

                obs_keys = ["agent_pos", "environment_state"]
                obs_shapes = [tuple(agent_pos.shape[1:]), tuple(env_state.shape[1:])]

                save_dict["obs_keys"] = np.array(obs_keys, dtype=object)
                save_dict["obs_shapes"] = np.array(obs_shapes, dtype=object)
                save_dict["states__agent_pos"] = agent_pos
                save_dict["states__environment_state"] = env_state
                save_dict["states"] = np.concatenate([agent_pos, env_state], axis=1).astype(np.float32)

        os.makedirs(os.path.dirname(self.traj_path), exist_ok=True)
        np.savez(self.traj_path, **save_dict)
        print(f"[PushT] Converted and saved trajectories to {self.traj_path}")

    # Minari materialization (original)
    def _materialize_from_minari(self):
        try:
            import minari
        except Exception as e:
            raise RuntimeError(
                f"Minari is required to fetch expert trajectories but is not available: {e}"
            )

        ds = None
        mapped_id = _map_env_to_minari_id(self.env_id)
        if mapped_id is not None:
            try:
                ds = minari.load_dataset(mapped_id, download=True)
                print(f"[Minari] Loaded via mapping: {self.env_id} -> {mapped_id}")
            except Exception as e:
                print(f"[Minari] Mapping load failed ({mapped_id}), fallback. Error: {e}")

        if ds is None:
            try:
                ds = minari.load_dataset(self.env_id, download=True)
                print(f"[Minari] Directly loaded dataset by id: {self.env_id}")
            except Exception:
                ds = None

        if ds is None:
            remote_ids = []
            if hasattr(minari, "list_remote_datasets"):
                try:
                    remote = minari.list_remote_datasets()
                    remote_ids = list(remote.keys()) if hasattr(remote, "keys") else list(remote)
                except Exception:
                    remote_ids = []
            candidate = None
            if self.env_id in remote_ids:
                candidate = self.env_id
            else:
                token = self.env_id.lower().split("_")[0]
                for rid in remote_ids:
                    if token in rid.lower():
                        candidate = rid
                        break
            if candidate is not None:
                ds = minari.load_dataset(candidate, download=True)
                print(f"[Minari] Downloaded and loaded dataset: {candidate}")

        if ds is None:
            raise FileNotFoundError(f"No Minari dataset found for env '{self.env_id}'.")

        episodes_iter = ds.iterate_episodes() if hasattr(ds, "iterate_episodes") else ds.episodes

        flat_states_list, actions_list, rewards_list = [], [], []
        flat_next_states_list, dones_list = [], []

        dict_mode = None
        per_key_states: Dict[str, list] = {}
        key_order: Optional[Tuple[str, ...]] = None
        leaf_shapes: Optional[List[Tuple[int, ...]]] = None
        kitchen_mode = _is_kitchen_env(self.env_id)

        for ep in episodes_iter:
            obs_seq = getattr(ep, "observations", None)
            acts_seq = getattr(ep, "actions", None)
            rews_seq = getattr(ep, "rewards", None)

            terms_seq = getattr(ep, "terminations", None)
            if terms_seq is None:
                terms_seq = getattr(ep, "terminated", None)
            trunc_seq = getattr(ep, "truncations", None)
            if trunc_seq is None:
                trunc_seq = getattr(ep, "truncated", None)

            if obs_seq is None or acts_seq is None or rews_seq is None:
                raise RuntimeError("Episode missing observations/actions/rewards.")

            n_steps = len(acts_seq)
            for t in range(n_steps):
                s_raw = self._slice_step(obs_seq, t)
                sp_raw = self._slice_step(obs_seq, t + 1)
                if kitchen_mode:
                    s_raw = self._canonicalize_kitchen_obs(s_raw)
                    sp_raw = self._canonicalize_kitchen_obs(sp_raw)

                a = acts_seq[t]
                r = rews_seq[t]
                term = bool(terms_seq[t]) if terms_seq is not None else False
                trunc = bool(trunc_seq[t]) if trunc_seq is not None else False
                done = term or trunc

                s_flat, s_leaf, key_order = self._flatten_obs_to_vec(
                    s_raw, key_order=key_order, sep=FLAT_KEY_SEP
                )
                sp_flat, _sp_leaf, _ = self._flatten_obs_to_vec(
                    sp_raw, key_order=key_order, sep=FLAT_KEY_SEP
                )

                flat_states_list.append(s_flat)
                flat_next_states_list.append(sp_flat)
                actions_list.append(self._format_action(a))
                rewards_list.append(np.float32(r))
                dones_list.append(done)

                if isinstance(s_raw, dict):
                    if dict_mode is None:
                        dict_mode = True
                    assert s_leaf is not None
                    if leaf_shapes is None:
                        leaf_shapes = [tuple(s_leaf[k].shape) for k in key_order]
                    for k in key_order:
                        per_key_states.setdefault(k, []).append(s_leaf[k])
                else:
                    if dict_mode is None:
                        dict_mode = False

        states_arr = np.asarray(flat_states_list, dtype=np.float32)
        next_states_arr = np.asarray(flat_next_states_list, dtype=np.float32)
        actions_arr = np.asarray(actions_list, dtype=np.float32)
        rewards_arr = np.asarray(rewards_list, dtype=np.float32)
        dones_arr = np.asarray(dones_list, dtype=np.bool_)

        save_dict = dict(
            states=states_arr,
            actions=actions_arr,
            rewards=rewards_arr,
            next_states=next_states_arr,
            dones=dones_arr,
        )

        if dict_mode:
            assert key_order is not None
            obs_keys = list(key_order)
            assert leaf_shapes is not None
            for k in obs_keys:
                save_dict[f"states__{k}"] = np.asarray(per_key_states[k], dtype=np.float32)
            save_dict["obs_keys"] = np.array(obs_keys, dtype=object)
            save_dict["obs_shapes"] = np.array(leaf_shapes, dtype=object)

        os.makedirs(os.path.dirname(self.traj_path), exist_ok=True)
        np.savez(self.traj_path, **save_dict)
        print(f"[Minari] Converted and saved full trajectories to {self.traj_path}")

    def _format_action(self, a) -> np.ndarray:
        if self.is_discrete:
            a_int = int(a) if not isinstance(a, (np.ndarray, list)) else int(np.asarray(a).item())
            vec = np.zeros(self.action_dim, dtype=np.float32)
            vec[a_int] = 1.0
            return vec
        else:
            vec = np.asarray(a, dtype=np.float32).reshape(-1)
            if vec.shape[0] != self.action_dim:
                vec = vec.reshape(self.action_dim)
            return vec

    def sample(self, batch_size: int):
        import random
        if self.num_samples == 0:
            raise RuntimeError("No samples available for training.")
        idx_pos = random.sample(range(self.num_samples), batch_size)
        idx = self.train_indices[idx_pos]
        obs_batch = torch.tensor(self.states[idx], dtype=torch.float32)
        act_batch = torch.tensor(self.actions[idx], dtype=torch.float32)
        return obs_batch, act_batch
