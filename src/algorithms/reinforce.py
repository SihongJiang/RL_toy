import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

def select_action(state, policy, device):
    state_tensor = torch.tensor(
        state,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)
    logits = policy(state_tensor)
    dist = Categorical(logits=logits)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action.item(), log_prob

def compute_returns(rewards, gamma):
    returns = []
    G = 0.0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(
        returns,
        dtype=torch.float32
    )
    return returns

def compute_advantages(
    returns, 
    use_baseline=True
):
    if use_baseline:
        baseline = returns.mean()
        advantages = returns - baseline
    else:
        advantages = returns
    return advantages
