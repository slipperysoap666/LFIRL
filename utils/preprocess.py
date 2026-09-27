from __future__ import annotations
from typing import List, Tuple, Sequence, Dict, Any
import numpy as np
import torch

__all__ = [
    "infer_traj_slices_from_offline",
    "to_torch",
    "one_hot_from_int",
    "make_action_tensor",
    "windowize_obs",
    "get_traj_slices_prefer_offline",
]


def infer_traj_slices_from_offline(offline) -> List[Tuple[int, int]]:
    """
    Infer episode (trajectory) slices [start, end_exclusive) from typical offline dataset fields.

    Priority:
      1) offline.traj_slices               -> iterable of (start, end_exclusive)
      2) offline.episode_ends or episode_starts
      3) scan terminals / timeouts / dones to split   (D4RL-style datasets commonly expose these)
      4) fallback: single slice over full array

    Notes:
      - "timeouts" correspond to truncations (time-limit ends), "terminals" to true terminations.
    """
    if hasattr(offline, "traj_slices") and offline.traj_slices:
        return [(int(s), int(e)) for (s, e) in offline.traj_slices]

    N = None
    # try to infer dataset length N
    if hasattr(offline, "actions"):
        N = len(offline.actions)
    elif hasattr(offline, "states"):
        N = len(offline.states)
    elif hasattr(offline, "states_dict"):
        some_key = next(iter(offline.states_dict))
        N = len(offline.states_dict[some_key])
    else:
        raise RuntimeError("Cannot infer dataset length from OfflineDataset.")

    if hasattr(offline, "episode_ends") and offline.episode_ends is not None:
        ends = np.asarray(offline.episode_ends, dtype=np.int64)
        starts = np.asarray(getattr(offline, "episode_starts", None))
        if starts is None or len(starts) != len(ends):
            starts = np.concatenate([[0], ends[:-1]])  # deduce starts
        return [(int(s), int(e)) for s, e in zip(starts, ends)]

    terminals = np.asarray(offline.terminals).astype(bool) if hasattr(offline, "terminals") else None
    timeouts = np.asarray(offline.timeouts).astype(bool) if hasattr(offline, "timeouts") else None
    dones = np.asarray(offline.dones).astype(bool) if hasattr(offline, "dones") else None

    end_mask = None
    if terminals is not None or timeouts is not None:
        tmask = terminals if terminals is not None else np.zeros(N, dtype=bool)
        omask = timeouts if timeouts is not None else np.zeros(N, dtype=bool)
        end_mask = (tmask | omask)
    elif dones is not None:
        end_mask = dones.astype(bool)

    if end_mask is not None:
        slices: List[Tuple[int, int]] = []
        start = 0
        for i in range(N):
            if end_mask[i]:
                slices.append((start, i + 1))
                start = i + 1
        if start < N:
            slices.append((start, N))
        return slices

    return [(0, N)]


def to_torch(x, device):
    """Convert to float32 tensor on the given device."""
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def one_hot_from_int(a_int: torch.Tensor, dim: int) -> torch.Tensor:
    """Map integer action(s) to one-hot vectors of length `dim`."""
    a_int = a_int.long().view(-1)
    out = torch.zeros((a_int.numel(), dim), device=a_int.device, dtype=torch.float32)
    out.scatter_(1, a_int.unsqueeze(1), 1.0)
    return out


def make_action_tensor(actions_row, action_dim: int, device: torch.device) -> torch.Tensor:
    """
    Convert a dataset action row to a 2D float tensor [1, A].

    Handles:
      - continuous (already float vector)
      - discrete stored as scalar int -> one-hot
    """
    a = torch.as_tensor(actions_row, device=device)
    if a.dtype in (torch.int32, torch.int64, torch.int16, torch.uint8) or a.dim() == 0:
        a = one_hot_from_int(a.view(1), action_dim)
    else:
        a = a.float().view(1, -1)
    return a


def windowize_obs(
        states_or_dict,
        t: int,
        start_idx: int,
        To: int,
        device: torch.device,
):
    """
    Build a length-To window ending at t (inclusive), respecting episode start.

    Shapes:
      - flat: returns Tensor [1, To, S]
      - dict: returns {k: Tensor [1, To, Sk]}

    Left padding: repeat the first valid frame of the episode.
    """
    left = max(start_idx, t - To + 1)
    pad = To - (t - left + 1)

    if isinstance(states_or_dict, dict):
        out = {}
        for k, arr in states_or_dict.items():
            seq = torch.as_tensor(arr[left: t + 1], dtype=torch.float32, device=device)  # [L, Sk]
            if pad > 0:
                first = torch.as_tensor(arr[left: left + 1], dtype=torch.float32, device=device)  # [1,Sk]
                pad_block = first.repeat(pad, 1)  # [pad, Sk]
                seq = torch.cat([pad_block, seq], dim=0)
            out[k] = seq.unsqueeze(0)  # [1, To, Sk]
        return out
    else:
        arr = states_or_dict
        seq = torch.as_tensor(arr[left: t + 1], dtype=torch.float32, device=device)  # [L, S]
        if pad > 0:
            first = torch.as_tensor(arr[left: left + 1], dtype=torch.float32, device=device)  # [1,S]
            pad_block = first.repeat(pad, 1)  # [pad, S]
            seq = torch.cat([pad_block, seq], dim=0)
        return seq.unsqueeze(0)  # [1, To, S]


def get_traj_slices_prefer_offline(offline) -> List[Tuple[int, int]]:
    """
    Return episode (start, end_exclusive) slices strictly preferring the segmentation
    carried by `offline` itself. Fallback to inference only if the dataset gives no
    explicit segmentation.

    Priority:
      1) offline.traj_slices (list/iterable of (start, end))
      2) offline.episode_starts & offline.episode_ends
      3) offline.trajectories (list-like). We try to derive contiguous slices either
         from per-trajectory (start,end) or by accumulating lengths (len(actions) or
         len(observations)/len(states)) assuming they are laid out in order.
      4) infer_traj_slices_from_offline(offline) using terminals/timeouts/dones
         per D4RL/Gymnasium conventions.  
                                           
    """
    if hasattr(offline, "traj_slices") and offline.traj_slices:
        return [(int(s), int(e)) for (s, e) in offline.traj_slices]

    starts = getattr(offline, "episode_starts", None)
    ends = getattr(offline, "episode_ends", None)
    if starts is not None and ends is not None:
        starts = np.asarray(starts, dtype=np.int64)
        ends = np.asarray(ends, dtype=np.int64)
        assert len(starts) == len(ends), "episode_starts and episode_ends length mismatch."
        return [(int(s), int(e)) for s, e in zip(starts, ends)]

    trajs = getattr(offline, "trajectories", None)
    if isinstance(trajs, Sequence) and len(trajs) > 0:
        slices: List[Tuple[int, int]] = []
        cursor = 0
        for traj in trajs:
            if isinstance(traj, dict):
                if "start" in traj and "end" in traj:
                    slices.append((int(traj["start"]), int(traj["end"])))
                    continue
                if "actions" in traj:
                    L = int(len(traj["actions"]))
                elif "observations" in traj:
                    L = int(len(traj["observations"]))
                elif "states" in traj:
                    L = int(len(traj["states"]))
                else:
                    raise RuntimeError("Cannot deduce trajectory length from 'trajectories' item.")
                slices.append((cursor, cursor + L))
                cursor += L
            else:
                if hasattr(traj, "start") and hasattr(traj, "end"):
                    slices.append((int(traj.start), int(traj.end)))
                elif hasattr(traj, "length"):
                    L = int(traj.length)
                    slices.append((cursor, cursor + L))
                    cursor += L
                else:
                    raise RuntimeError("Unrecognized trajectory object: missing start/end/length.")
        return slices

    return infer_traj_slices_from_offline(offline)
