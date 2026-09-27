# LFIRL

LFIRL is an offline inverse reinforcement learning pipeline built around a pretrained diffusion or flow-matching policy. The algorithm first prepares expert demonstrations, pretrains or loads the policy model, then learns the Q, V, calibrated value, and finally reward networks used by the LFIRL reward model.

## Repository Layout

```text
.
|-- train_lfirl.py                  # Main LFIRL training entry point
|-- evaluate.py                     # Reward ranking evaluation
|-- environment.yml                 # Conda environment specification
|-- environments/                   # Per-environment YAML configs
|-- datasets/                       # Offline dataset loading and sequence windows
|-- models/                         # Diffusion policy, Q, V, and reward networks
|-- policy_training/                # Diffusion/flow policy pretraining
`-- utils/                          # Environment, logging, preprocessing, and evaluation helpers
```

Available configs:

```text
environments/AdroitHandPen-v1.yml
environments/FrankaKitchen-v1.yml
environments/PointMaze_Large-v3.yml
environments/PointMaze_Medium-v3.yml
environments/PointMaze_UMaze-v3.yml
environments/PushT-v0.yml
```

## Installation

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate LFIRL
```

The environment uses Python 3.9.21 and installs CUDA 12.8 PyTorch wheels through pip. It also includes conda packages commonly needed by `mujoco-py` on systems without sudo access, including `glew`, `glfw`, `mesalib`, `libosmesa`, and `patchelf`.

## Training

Run LFIRL with one of the environment configs:

```bash
python train_lfirl.py --cfg environments/PointMaze_UMaze-v3.yml
```

Each config controls the environment id, policy pretraining settings, Q/b/V/reward network sizes, LFIRL stage passes, and output paths.

By default, paths are relative to the project root:

```text
pretrained_policy/    # Cached pretrained diffusion or flow policy checkpoints
expert_data/          # Materialized expert trajectory archives
results/              # Logs and learned model checkpoints
```

If a pretrained policy checkpoint already exists in `pretrained_policy/`, the training script loads it. Otherwise, it pretrains the policy from the expert data and saves the checkpoint before running the LFIRL stages.

## Evaluation

After training, evaluate a learned reward network by ranking expert trajectories against poor generated trajectories:

```bash
python evaluate.py \
  --reward_path results/<env>/<run>/r_net.pt \
  --cfg environments/PushT-v0.yml \
  --num_traj 200 \
  --device auto
```

## Data Notes

The dataset loader materializes expert demonstrations into `expert_data/` as `.npz` archives. For supported Minari/D4RL-style environments, data is loaded through Minari. For PushT, the loader can download the Diffusion Policy PushT archive, extract it, and convert the zarr data into the same trajectory format.

Generated files under `expert_data/`, `pretrained_policy/`, and `results/` are runtime artifacts and do not need to be committed unless you intentionally want to share checkpoints or cached data.


## Quick Start

```bash
conda env create -f environment.yml
conda activate LFIRL
python train_lfirl.py --cfg environments/PointMaze_UMaze-v3.yml
python evaluate.py --reward_path results/PointMaze_UMaze-v3/r_net.pt --cfg environments/PointMaze_UMaze-v3.yml
```

Adjust the `--reward_path` to the actual checkpoint path printed or saved by your run.
