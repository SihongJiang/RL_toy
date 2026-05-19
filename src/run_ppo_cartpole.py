import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from datetime import datetime
from torch.distributions import Categorical

from utils.seed import set_seed
from envs.make_env import make_env
from utils.plot import plot_training_curves
from networks.mlp import PolicyNetwork, ValueNetwork
from buffers.rollout_buffer import RolloutBuffer
from algorithms.ppo import (
    collect_rollout,
    get_value,
    compute_gae,
    evaluate_actions,
)
from trainers.train_ppo import train_ppo
import yaml

import shutil

PROJECT_ROOT = "/home/zhangsihong/Projects/rl_toy"

run_name = datetime.now().strftime("ppo_cartpole_%Y%m%d_%H%M%S")
output_dir = os.path.join(PROJECT_ROOT, "tmp", run_name)
os.makedirs(output_dir, exist_ok=True)
print(f"[INFO] Output directory: {output_dir}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

config_path = os.path.join(PROJECT_ROOT, "configs", "ppo_cartpole.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
shutil.copy(
    config_path,
    os.path.join(output_dir, "config.yaml")
)
print(f"[INFO] Config saved to: {os.path.join(output_dir, 'config.yaml')}")


set_seed(config["seed"])
# 创建环境
env = make_env(config["env"]["name"])
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

print("state dim:", state_dim)
print("action dim:", action_dim)

hidden_dim = config["model"]["hidden_dim"]
# 策略实例化
actor = PolicyNetwork(
    state_dim=state_dim,
    hidden_dim=hidden_dim,
    action_dim=action_dim
).to(device)
# critic实例化
critic = ValueNetwork(
    state_dim=state_dim,
    hidden_dim=hidden_dim
).to(device)

# actor和critic的优化器定义
actor_optimizer = optim.Adam(
    actor.parameters(),
    lr =config["optimizer"]["actor_lr"]
)
critic_optimizer = optim.Adam(
    critic.parameters(),
    lr=config["optimizer"]["critic_lr"]
)

train_cfg = config["train"]

history = train_ppo(
    env=env,
    actor=actor,
    critic=critic,
    actor_optimizer=actor_optimizer,
    critic_optimizer=critic_optimizer,
    device=device,
    **train_cfg
)

metadata = {
    "env_name": config["env"]["name"],
    "seed": config["seed"],
    "device": str(device),
    "state_dim": state_dim,
    "action_dim": action_dim,
    "output_dir": output_dir,
}
metadata_path = os.path.join(output_dir, "metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"[INFO] Metadata saved to: {metadata_path}")

import json
history_path = os.path.join(output_dir, "history.json")
with open(history_path, "w") as f:
    json.dump(
        {
            k: [float(x) for x in v]
            for k, v in history.items()
        },
        f,
        indent=2
    )
print(f"[INFO] History saved to: {history_path}")

plot_training_curves(history, output_dir)