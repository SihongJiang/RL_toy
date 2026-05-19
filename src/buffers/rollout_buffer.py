import numpy as np
import torch


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.old_log_probs = []
        self.values = []

    def add(self, state, action, reward, done, old_log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.old_log_probs.append(old_log_prob)
        self.values.append(value)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.old_log_probs.clear()
        self.values.clear()

    def to_tensors(self, device):
        states = torch.tensor(
            np.array(self.states),
            dtype=torch.float32,
            device=device
        )
        actions = torch.tensor(
            self.actions,
            dtype=torch.long,
            device=device
        )
        rewards = torch.tensor(
            self.rewards,
            dtype=torch.float32,
            device=device
        )
        dones = torch.tensor(
            self.dones,
            dtype=torch.float32,
            device=device
        )
        old_log_probs = torch.stack(self.old_log_probs).to(device)
        values = torch.stack(self.values).to(device)

        return states, actions, rewards, dones, old_log_probs, values