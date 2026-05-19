import torch
from torch.distributions import Categorical

from buffers.rollout_buffer import RolloutBuffer


def select_action_ppo(state, actor, critic, device):
    state_tensor = torch.tensor(
        state,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    with torch.no_grad():
        logits = actor(state_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample()
        old_log_prob = dist.log_prob(action)
        value = critic(state_tensor)

    return (
        action.item(),
        old_log_prob.squeeze(0),
        value.squeeze(-1).squeeze(0)
    )


def collect_rollout(env, state, actor, critic, device, rollout_steps=2048):
    buffer = RolloutBuffer()
    episode_rewards = []
    episode_reward = 0.0

    for _ in range(rollout_steps):
        action, old_log_prob, value = select_action_ppo(
            state=state,
            actor=actor,
            critic=critic,
            device=device
        )

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        buffer.add(
            state=state,
            action=action,
            reward=reward,
            done=done,
            old_log_prob=old_log_prob,
            value=value
        )

        episode_reward += reward
        state = next_state

        if done:
            episode_rewards.append(episode_reward)
            episode_reward = 0.0
            state, info = env.reset()

    return buffer, state, episode_rewards


def get_value(state, critic, device):
    state_tensor = torch.tensor(
        state,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    with torch.no_grad():
        value = critic(state_tensor)

    return value.squeeze()


def compute_gae(
    rewards,
    dones,
    values,
    last_value,
    gamma=0.99,
    gae_lambda=0.95
):
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    next_value = last_value

    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[t] = gae
        next_value = values[t]

    returns = advantages + values
    return advantages.detach(), returns.detach()


def evaluate_actions(states, actions, actor, critic):
    logits = actor(states)
    dist = Categorical(logits=logits)

    new_log_probs = dist.log_prob(actions)
    entropy = dist.entropy()
    values = critic(states).squeeze(-1)

    return new_log_probs, entropy, values