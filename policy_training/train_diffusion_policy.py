import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.diffusion_model import DiffusionPolicyTimeseries, FlowMatchingPolicyTimeseries
from datasets.dataset import OfflineDataset
from datasets.dataset_seq import SequenceDataset
from utils.utils import (
    to_device_obs,
    evaluate_policy_receding,
)


def sanitize_env_id(env_id: str) -> str:
    return env_id.replace("/", "_").replace(":", "_")


def resolve_policy_backend(policy_cfg):
    backend = str(policy_cfg.get("diffusion_model", "ddpm")).strip().lower().replace("-", "_")
    aliases = {
        "ddpm": "ddpm",
        "flow": "flow_matching",
        "flowmatch": "flow_matching",
        "flow_matching": "flow_matching",
    }
    backend = aliases.get(backend, backend)
    if backend not in {"ddpm", "flow_matching"}:
        raise ValueError(
            f"Unsupported policy.diffusion_model='{backend}'. "
            "Expected one of: ddpm, flow_matching."
        )
    return backend


def build_diffusion_policy_model(
    *,
    obs_dim,
    action_dim,
    device,
    cfg=None,
    backend=None,
    **model_kwargs,
):
    common_kwargs = dict(
        obs_dim=obs_dim,
        action_dim=action_dim,
    )

    if cfg is not None:
        p = cfg["policy"]
        if backend is None:
            backend = resolve_policy_backend(p)
        common_kwargs.update(
            dict(
                horizon=int(p["pred_horizon"]),
                n_obs_steps=int(p["obs_horizon"]),
                n_action_steps=int(p["pred_horizon"]),
                num_train_timesteps=int(p["timesteps"]),
                num_inference_steps=int(p["ddim_steps"]),
                obs_as_local_cond=False,
                obs_as_global_cond=True,
                pred_action_steps_only=True,
            )
        )

    common_kwargs.update(model_kwargs)

    if backend is None:
        raise ValueError("build_diffusion_policy_model requires either cfg or backend.")

    model_cls = DiffusionPolicyTimeseries if backend == "ddpm" else FlowMatchingPolicyTimeseries
    return model_cls(**common_kwargs).to(device)


def get_policy_checkpoint_candidates(env_id, pretrained_dir, policy_cfg):
    env_tag = sanitize_env_id(env_id)
    backend = resolve_policy_backend(policy_cfg)

    preferred = os.path.join(pretrained_dir, f"{env_tag}_{backend}_policy.pt")
    candidates = [preferred]

    if backend == "ddpm":
        legacy = os.path.join(pretrained_dir, f"{env_tag}_policy.pt")
        if legacy not in candidates:
            candidates.append(legacy)

    return preferred, candidates


def train_diffusion_policy(env, cfg, device):
    """
    Pretrain a Diffusion Policy on expert data (Minari) and return the model.

    """
    p = cfg["policy"]
    backend = resolve_policy_backend(p)

    offline = OfflineDataset(
        cfg["env_id"],
        env=env,
        num_expert=p["num_expert"],
        seed=p.get("seed", None),
    )
    action_dim = offline.action_dim

    if getattr(offline, "states_dict", None) is not None:
        state_keys = sorted(offline.states_dict.keys())
        obs_arr = np.concatenate(
            [offline.states_dict[k] for k in state_keys],
            axis=-1,
        )  # [N, Do]
        obs_dim = obs_arr.shape[-1]
        print(f"[Dims] dict-obs keys={state_keys}, obs_dim={obs_dim}")
    else:
        # flat obs
        obs_arr = offline.states  # [N, Do]
        obs_dim = obs_arr.shape[-1]
        print(f"[Dims] flat-obs, obs_dim={obs_dim}")

    To = int(p["obs_horizon"])
    Tp = int(p["pred_horizon"])
    print(
        f"[Dims] action_dim={action_dim} | "
        f"To={To}, Tp={Tp}, Ta={p['action_horizon']}"
    )

    seq_ds = SequenceDataset(
        states=None if getattr(offline, "states_dict", None) is not None else offline.states,
        actions=offline.actions,
        dones=offline.dones,
        obs_horizon=To,
        pred_horizon=Tp,
        indices_subset=offline.train_indices,
        states_dict=getattr(offline, "states_dict", None),
    )
    g = torch.Generator().manual_seed(int(p["seed"]))
    loader = DataLoader(
        seq_ds,
        batch_size=int(p["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=int(p["num_workers"]),
        pin_memory=bool(p["pin_memory"]),
        generator=g,
    )

    model = build_diffusion_policy_model(
        obs_dim=obs_dim,
        action_dim=action_dim,
        backend=backend,
        device=device,
        horizon=Tp,
        n_obs_steps=To,
        n_action_steps=Tp,
        num_train_timesteps=int(p["timesteps"]),
        num_inference_steps=int(p["ddim_steps"]),
        obs_as_local_cond=False,
        obs_as_global_cond=True,
        pred_action_steps_only=True,
    )

    data_for_fit = {
        "obs": obs_arr,              # [N, obs_dim]
        "action": offline.actions,   # [N, action_dim]
    }
    model.normalizer.fit(
        data_for_fit,
        last_n_dims=1,
        mode="limits",
        output_min=-1.0,
        output_max=1.0,
    )
    print("[Norm] Fitted LinearNormalizer on offline obs/actions.")

    optimizer = optim.AdamW(model.parameters(), lr=float(p["lr"]))

    epochs = int(p["epochs"])
    eval_interval = int(p["eval_interval"])
    eval_episodes = int(p["eval_episodes"])

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0

        for obs_seq, act_seq in loader:
            # act_seq: [B, Tp, A]
            obs_seq = to_device_obs(obs_seq, device)
            act_seq = act_seq.to(device)

            obs_flat = model.flatten_obs(obs_seq)  # [B, To, obs_dim]

            batch = {
                "obs": obs_flat,
                "action": act_seq,
            }

            loss = model.compute_loss(batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item()

        if epoch % eval_interval == 0:
            avg_loss = running / max(1, len(loader))
            model.eval()
            eval_metric = evaluate_policy_receding(
                model=model,
                env=env,
                episodes=eval_episodes,
                obs_horizon=To,
                pred_horizon=Tp,
                action_horizon=int(p["action_horizon"]),
                device=device,
                base_seed=int(p["seed"]),
            )
            print(f"[Epoch {epoch}] TrainLoss: {avg_loss:.4e} | Eval metric: {eval_metric:.3f}")

    return model
