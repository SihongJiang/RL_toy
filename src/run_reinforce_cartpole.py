import os
from datetime import datetime
import json
import shutil

import torch
import torch.optim as optim
import yaml

from utils.seed import set_seed
from envs.make_env import make_env
from networks.mlp import PolicyNetwork
from trainers.train_reinforce import train_reinforce
from utils.plot import plot_reinforce_training_curves


PROJECT_ROOT = "/home/zhangsihong/Projects/rl_toy"

run_name = datetime.now().strftime("reinforce_cartpole_%Y%m%d_%H%M%S")
output_dir = os.path.join(PROJECT_ROOT, "tmp", run_name)
os.makedirs(output_dir, exist_ok=True)

print(f"[INFO] Output directory: {output_dir}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {device}")

config_path = os.path.join(PROJECT_ROOT, "configs", "reinforce_cartpole.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

shutil.copy(
    config_path,
    os.path.join(output_dir, "config.yaml")
)

print(f"[INFO] Config saved to: {os.path.join(output_dir, 'config.yaml')}")

set_seed(config["seed"])

env = make_env(config["env"]["name"])

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

print(f"[INFO] State dim: {state_dim}")
print(f"[INFO] Action dim: {action_dim}")

hidden_dim = config["model"]["hidden_dim"]

policy = PolicyNetwork(
    state_dim=state_dim,
    hidden_dim=hidden_dim,
    action_dim=action_dim,
).to(device)

optimizer = optim.Adam(
    policy.parameters(),
    lr=config["optimizer"]["lr"],
)

train_cfg = config["train"]

history = train_reinforce(
    env=env,
    policy=policy,
    optimizer=optimizer,
    device=device,
    **train_cfg,
)

metadata = {
    "algorithm": "REINFORCE",
    "env_name": config["env"]["name"],
    "seed": int(config["seed"]),
    "device": str(device),
    "state_dim": int(state_dim),
    "action_dim": int(action_dim),
    "output_dir": str(output_dir),
}

metadata_path = os.path.join(output_dir, "metadata.json")

with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"[INFO] Metadata saved to: {metadata_path}")

history_path = os.path.join(output_dir, "history.json")

with open(history_path, "w") as f:
    json.dump(
        {
            k: [float(x) for x in v]
            for k, v in history.items()
        },
        f,
        indent=2,
    )

print(f"[INFO] History saved to: {history_path}")

plot_reinforce_training_curves(history, output_dir)