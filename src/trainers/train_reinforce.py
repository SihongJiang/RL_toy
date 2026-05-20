import numpy as np
import torch

from algorithms.reinforce import (
    select_action,
    compute_returns,
    compute_advantages
)

def train_reinforce(
    env,
    policy,
    optimizer,
    device,
    num_episodes=1000,
    gamma=0.99,
    use_baseline=True,
    rolling_window=50,
):
    rewards_history = []
    rolling_return_variance = []
    rolling_reward_mean_history = []
    raw_returns_variance_history = []
    advantage_variance_history = []
    policy_loss_variance_history=[]
    gradient_norm_history = []

    for episode in range(num_episodes):
        state, info = env.reset()
        log_probs = []
        rewards = []
        done = False
        while not done:
            action, log_prob= select_action(
                state=state,
                policy=policy,
                device=device
            )
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
        returns = compute_returns(rewards=rewards, gamma=gamma).to(device)
        raw_return_variance = returns.var(unbiased=False).item()
        raw_returns_variance_history.append(raw_return_variance)
        advantages = compute_advantages(
            returns=returns,
            use_baseline=use_baseline
        ).to(device)
        advantage_variance = advantages.var(unbiased=False).item()
        advantage_variance_history.append(advantage_variance)
        policy_loss = []
        policy_loss_values = []
        for log_prob, advantage in zip(log_probs, advantages):
            loss_item = (-log_prob * advantage)
            policy_loss.append(loss_item)
            policy_loss_values.append(loss_item.item())
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss_variance = np.var(policy_loss_values)
        policy_loss_variance_history.append(policy_loss_variance)

        optimizer.zero_grad()
        policy_loss.backward()
        total_norm = 0.0
        for p in policy.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        gradient_norm_history.append(total_norm)
        optimizer.step()
        episode_reward = sum(rewards)
        rewards_history.append(episode_reward)
        if episode % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(
                f"Episode {episode}, "
                f"Reward: {episode_reward:.1f}, "
                f"Avg Reward: {avg_reward:.1f} "
            )
        rolling_return_variance.append(
            np.var(rewards_history[-rolling_window:])
        )
        rolling_reward_mean_history.append(
            np.mean(rewards_history[-rolling_window:])
        )
    history = {
        "rewards_history": rewards_history,
        "rolling_reward_mean_history": rolling_reward_mean_history,
        "rolling_return_variance": rolling_return_variance,
        "raw_returns_variance_history": raw_returns_variance_history,
        "advantage_variance_history": advantage_variance_history,
        "policy_loss_variance_history": policy_loss_variance_history,
        "gradient_norm_history": gradient_norm_history,
    }
    return history