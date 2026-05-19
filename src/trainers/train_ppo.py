import numpy as np
import torch
import torch.nn as nn

from algorithms.ppo import (
    collect_rollout,
    get_value,
    compute_gae,
    evaluate_actions,
)

def train_ppo(
    env,
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    device,
    num_updates=500,
    rollout_steps=2048,
    minibatch_size=64,
    ppo_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
    max_grad_norm=0.5
):
    rewards_history=[]
    actor_loss_history=[]
    critic_loss_history=[]
    entropy_history=[]
    approx_kl_history=[]
    clip_fraction_history=[]
    state, info = env.reset()
    for update in range(num_updates):
        # collect rollout
        buffer, state, episode_rewards = collect_rollout(
            env=env,
            state=state,
            actor=actor,
            critic=critic,
            device=device,
            rollout_steps=rollout_steps
        )
        states, actions, rewards, dones, old_log_probs, values = buffer.to_tensors(device)
        last_value = get_value(
            state=state,
            critic=critic,
            device=device
        )
        advantages, returns = compute_gae(
            rewards=rewards, dones=dones, values=values,
            last_value=last_value, gamma=gamma, gae_lambda=gae_lambda
        )
        # advantages normalization
        advantages = (advantages - advantages.mean())/(
            advantages.std(unbiased=False)+1e-8
        )
        # ppo update
        batch_size = states.shape[0]
        for epoch in range(ppo_epochs):
            indices = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, minibatch_size):
                end = start+minibatch_size
                mb_idx = indices[start:end]
                batch_states = states[mb_idx]
                batch_actions = actions[mb_idx]
                batch_old_log_probs = old_log_probs[mb_idx]
                batch_advantages = advantages[mb_idx]
                batch_returns = returns[mb_idx]
                # new actions, states, ……
                new_log_probs, entropy, new_values = evaluate_actions(
                    states=batch_states,
                    actions=batch_actions,
                    actor=actor,
                    critic=critic
                )
                # ppo clipped actor loss
                ratio = torch.exp(new_log_probs-batch_old_log_probs)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0-clip_eps,
                    1.0+clip_eps
                )*batch_advantages
                actor_loss = -torch.min(surr1,surr2).mean()
                # critic loss
                critic_loss = nn.functional.mse_loss(new_values,batch_returns)
                # entropy loss
                entropy_loss = -entropy.mean()

                total_loss = actor_loss + vf_coef*critic_loss + ent_coef*entropy_loss

                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
                actor_optimizer.step()
                critic_optimizer.step()

                #============ logging ===========
                actor_loss_history.append(actor_loss.item())
                critic_loss_history.append(critic_loss.item())
                entropy_history.append(entropy.mean().item())
                with torch.no_grad():
                    approx_kl = (batch_old_log_probs-new_log_probs).mean().item()
                    clip_fraction = ((ratio-1.0).abs()>clip_eps).float().mean().item()
                approx_kl_history.append(approx_kl), clip_fraction_history.append(clip_fraction)

        #========== reward logging ===========
        if len(episode_rewards) > 0:
            avg_episode_reward = np.mean(episode_rewards)
            rewards_history.append(avg_episode_reward)
            if update % 10 == 0:
                recent_avg = np.mean(rewards_history[-10:])
                print(
                    f"Update {update}, "
                    f"Avg Episode Reward: {avg_episode_reward:.1f}, "
                    f"Recent Avg Reward: {recent_avg:.1f}, "
                    f"Actor Loss: {np.mean(actor_loss_history[-10:]):.4f}, "
                    f"Critic Loss: {np.mean(critic_loss_history[-10:]):.4f}, "
                    f"Entropy: {np.mean(entropy_history[-10:]):.4f}"
                )    
    return {
        "rewards_history": rewards_history,
        "actor_loss_history": actor_loss_history,
        "critic_loss_history": critic_loss_history,
        "entropy_history": entropy_history,
        "approx_kl_history": approx_kl_history,
        "clip_fraction_history": clip_fraction_history,
    }