from typing import Optional, Dict, Tuple, List
import numpy as np
import torch

Array = np.ndarray
Tensor = torch.Tensor


class SequenceDataset(torch.utils.data.Dataset):
    """
    Create sliding windows over transitions **without crossing episode boundaries**.
      - flat: states [N, S]            -> obs_seq: Tensor [To, S]
      - dict: states_dict {k: [N, Sk]} -> obs_seq: Dict[str, Tensor [To, Sk]]
    Target: action sequence [Tp, A] (indices t+1 .. t+Tp)
    """

    def __init__(self,
                 states: Optional[Array],
                 actions: Array,
                 dones: Array,
                 obs_horizon: int,
                 pred_horizon: int,
                 indices_subset: Optional[Array] = None,
                 states_dict: Optional[Dict[str, Array]] = None):
        assert (states is not None) or (states_dict is not None), \
            "Provide either `states` (flat) or `states_dict` (dict of arrays)."

        self.is_dict = states_dict is not None
        self.states = states
        self.states_dict = states_dict
        self.actions = actions
        self.dones = dones.astype(bool)
        self.To = obs_horizon
        self.Tp = pred_horizon

        # episode ranges
        self.ranges: List[Tuple[int, int]] = []
        start = 0
        N = len(dones)
        for i in range(N):
            if self.dones[i]:
                self.ranges.append((start, i))
                start = i + 1
        if start < N:
            self.ranges.append((start, N - 1))

        # valid centers
        self.samples: List[Tuple[int, int, int]] = []
        for (s, e) in self.ranges:
            t_min = s + (self.To - 1)
            t_max = e - self.Tp
            if t_max >= t_min:
                self.samples.extend([(t, s, e) for t in range(t_min, t_max + 1)])

        # optional subset filtering (by transition index)
        if indices_subset is not None:
            mask = set(indices_subset.tolist())
            self.samples = [u for u in self.samples if u[0] in mask]

    def __len__(self):
        return len(self.samples)

    def _get_obs_window_flat(self, t: int) -> Tensor:
        obs_start = t - (self.To - 1)
        obs_end = t
        seq = self.states[obs_start:obs_end + 1]  # [To, S]
        return torch.tensor(seq, dtype=torch.float32)

    def _get_obs_window_dict(self, t: int) -> Dict[str, Tensor]:
        obs_start = t - (self.To - 1)
        obs_end = t
        out: Dict[str, Tensor] = {}
        for k, arr in self.states_dict.items():
            seq = arr[obs_start:obs_end + 1]  # [To, Sk]
            out[k] = torch.tensor(seq, dtype=torch.float32)
        return out

    def __getitem__(self, idx):
        t, s, e = self.samples[idx]
        act_start = t + 1
        act_end = t + self.Tp
        act_seq = self.actions[act_start:act_end + 1]  # [Tp, A]
        act_t = torch.tensor(act_seq, dtype=torch.float32)

        if self.is_dict:
            obs = self._get_obs_window_dict(t)
        else:
            obs = self._get_obs_window_flat(t)
        return obs, act_t
