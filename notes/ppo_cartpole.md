# PPO+Cartpole Notes

## Running PPO + CartPole

Run PPO training with:
```bash
bash scripts/run_ppo_cartpole.sh
```
Each run automatically creates an isolated experiment directory under:
```text
tmp/ppo_cartpole_YYYYMMDD_HHMMSS/
```
where the folder name records the experiment start time (year-month-day_hour-minute-second).
Each experiment directory contains:
```text
config.yaml           # experiment hyperparameters
metadata.json         # environment / device / run information
history.json          # raw logged training metrics
training_curves.png   # visualization of training curves
```
## Why PPO: From REINFORCE to PPO

REINFORCE is the most basic policy gradient method. It directly increases the probability of actions that lead to high returns and decreases the probability of actions that lead to low returns. Its core idea is simple, but in practice it suffers from:
- high variance
- unstable updates
- and poor sample efficiency

PPO can be understood as a series of practical modifications built on top of this basic policy-gradient idea.

### 1. From Return to Advantage: Reducing Variance

In vanilla REINFORCE, the policy is updated using the Monte Carlo return $G_t$. However, $G_t$ can have high variance because it depends on the full sampled trajectory.

PPO instead uses an advantage estimate:
$$
A_t = Q(s_t, a_t) - V(s_t)
$$

The advantage measures whether an action's expected discounted reward $Q(s_t, a_t)$ is better or worse than the expected value of the current state $V(s_t)$. This makes the policy update more stable by reducing the total variance of policy gradient.

（详细的REINFORCE策略梯度的方差推导将在后续放到REINFORCE的notes中。）

In this project, advantages are estimated with GAE.

### 2. Actor-Critic: Learning a Value Baseline

PPO introduces an actor-critic structure to evaluate $V(s_t)$ every time:
```text
actor  -> outputs action probabilities
critic -> estimates state value V(s)
```
The critic provides a learned baseline $V(s_t)$, which is independent of the selected action. Subtracting this baseline does not change the expectation of the policy gradient, but can significantly reduce its variance.

Intuitively, the actor can be viewed as a student who tries to perform well on every problem (state), while the critic acts like a coach who estimates the overall difficulty of the current problem. After removing the effect of problem difficulty, the remaining quantity — the advantage — reflects whether the chosen action performed better or worse than expected under the current state.

This makes policy updates much more stable than directly optimizing with raw trajectory returns.

### 3. GAE: Advantage Estimation with Bias-Variance Tradeoff
Generalized Advantage Estimation (GAE) is used to compute smoother and more reliable advantage estimates.

GAE combines multi-step temporal-difference errors:
$$
\delta_t = r_t + \gamma * V(s_{t+1}) - V(s_t)
$$
and accumulates them with a decay factor:
$$
A_t^{\text{GAE}(\gamma,\lambda)} = \delta_t + \gamma * \lambda * \delta_{t+1} + ...
$$

$$
A_t^{\text{GAE}(\gamma,\lambda)}
=
\sum_{l=0}^{\infty}
(\gamma \lambda)^l
\delta_{t+l}
$$
The parameter $\lambda$ controls the bias-variance tradeoff. When $\lambda = 0$, GAE reduces to a one-step TD estimate, which usually has lower variance but higher bias because it relies heavily on the learned value function. When $\lambda = 1$, GAE becomes closer to a Monte Carlo-style advantage estimate, which has lower bias but higher variance because it depends more on long-horizon sampled rewards.

### 4. Policy Ratio: Measuring Policy Change

PPO measures how much the new policy has changed compared with the old policy that collected the rollout data using the policy ratio:

$$
r_t(\theta)
=
\frac{
\pi_\theta(a_t \mid s_t)
}{
\pi_{\theta_{\text{old}}}(a_t \mid s_t)
}
$$

Here, the “old policy” refers to the policy used to generate the rollout data, while the “new policy” refers to the current policy being optimized.

If $r_t(\theta) > 1$, the new policy assigns a higher probability to the sampled action than the old policy. If $r_t(\theta) < 1$, the new policy assigns a lower probability to that action.

The policy ratio allows PPO to reuse rollout data for multiple optimization epochs while still tracking how far the updated policy has moved away from the behavior policy that generated the data.

More importantly, the ratio provides a direct measurement of policy change, which later enables PPO to constrain overly large updates through the clipped objective.

### 5. Clipped Objective: Preventing Overly Large Policy Updates (Approximate Trust Region)

A major problem in policy gradient methods is that one update can change the policy too much and destroy previously learned behavior. PPO addresses this with the clipped surrogate objective:

$$
L_{clip} = \min(
    r_t(\theta) * A_t,
    clip(r_t(\theta), 1 - \epsilon, 1 + \epsilon) * A_t
)
$$

The clipping mechanism limits the effective policy update when the new policy becomes too different from the old policy. 

When $A_t > 0$, the clipped objective prevents the policy ratio from becoming much larger than $(1 + \epsilon)$, limiting overly aggressive probability increases for beneficial actions.

When $A_t < 0$, the clipped objective prevents the policy ratio from becoming much smaller than $(1 - \epsilon)$, limiting overly aggressive probability decreases for harmful actions.

This makes PPO behave like an approximate trust-region method: it encourages improvement, but discourages destructive policy shifts.

### 6. Entropy Regularization: Encouraging Exploration

To prevent the policy from becoming too deterministic at an early stage, which can reduce exploration and trap the agent in a suboptimal behavior pattern, PPO often adds an entropy regularization term to the total training loss \(L_{\text{total}}\):

$$
L_{\text{entropy}} = - \text{entropy}(\pi_\theta(. \mid s_t))
$$

This term encourages the policy distribution to maintain some uncertainty during training. Higher entropy means the action probabilities are more evenly distributed, indicating that the policy is still exploring and remains less certain about which action is optimal. Lower entropy means the policy becomes more confident and deterministic, assigning much higher probability to a small subset of actions.

In CartPole, entropy usually decreases as training progresses, because the agent gradually learns a stable balancing strategy and no longer needs to explore actions as randomly as in the early training stage.

### 7. Rollout Buffer and Minibatch Updates: Storing On-Policy Training Data for limited data reusing

Like REINFORCE, PPO is still an on-policy algorithm: the sampled trajectories must be generated by the current policy being trained. PPO improves sample efficiency by reusing the collected rollout data for multiple optimization epochs.

To achieve this, PPO stores one rollout batch in a rollout buffer. In this project, the buffer records:
- states
- actions
- rewards
- dones
- old_log_probs
- values

In particular, `old_log_probs` are used to compute the policy ratio, while `values` are used to compute GAE advantages and returns.

More specifically, PPO first treats the current policy as the “old policy” and uses it to collect a rollout batch from the environment. The collected data are then fixed inside the rollout buffer.

After the rollout is collected, PPO performs several minibatch optimization epochs on this same batch of data. During these optimization steps, the policy parameters are continuously updated, but the ratio and GAE computations still rely on the rollout data generated by the old policy.

After all PPO minibatch epochs finish, the rollout buffer is discarded. The updated policy is then treated as the new “old policy”, and a new rollout batch is collected from the environment.

This mechanism improves sample efficiency compared with using each trajectory only once, while the clipped objective prevents the policy from drifting too far away from the policy that originally generated the rollout data.

## PPO Training Pipeline

This section connects the PPO training logic with the actual code structure in this repository.

The full training entry point is:
```text
src/run_ppo_cartpole.py
```
This script loads the configuration, creates the environment, builds the actor and critic networks, initializes optimizers, calls the PPO trainer, and saves the experiment outputs.

### 0. Overall Training Flow
The overall training pipeline is:
```text
src/run_ppo_cartpole.py
        |
        |-- load configs/ppo_cartpole.yaml
        |-- create CartPole-v1 environment
        |-- build actor and critic networks
        |-- create optimizers
        |
        v
src/trainers/train_ppo.py
        |
        |-- collect rollout
        |-- compute GAE
        |-- normalize advantages
        |-- run minibatch PPO updates
        |-- record training metrics
        |
        v
src/utils/plot.py
        |
        |-- save training_curves.png
```
In short:
```text
configuration
-> environment
-> actor/critic networks
-> rollout collection
-> advantage estimation
-> PPO optimization
-> logging and visualization
```
Run PPO + CartPole with:
```
bash scripts/run_ppo_cartpole.sh
```
Each run creates an isolated output directory under:
```
tmp/ppo_cartpole_YYYYMMDD_HHMMSS/
```
The timestamp in the folder name records when the experiment was started.

Each output directory contains:
```
config.yaml           # copied experiment configuration
metadata.json         # environment, device, and run information
history.json          # raw training metrics
training_curves.png   # plotted training curves
```
A selected complete run is also stored under:
```
results/ppo_cartpole/main_run/
```

### 1. Load Configuration
The experiment hyperparameters are stored in:
```text
configs/ppo_cartpole.yaml
```
The configuration includes:

- environment name
- random seed
- hidden dimension
- actor / critic learning rates
- rollout steps
- PPO epochs
- minibatch size
- discount factor
- GAE lambda
- clipping coefficient
- entropy coefficient
- value loss coefficient

In `src/run_ppo_cartpole.py`, the config is loaded before training starts and is copied into the output directory as `config.yaml`.

### 2. Create Environment
The environment is created by:
```
src/envs/make_env.py
```
In this project, the current environment is:
```
CartPole-v1
```
The environment provides:
- state observations
- discrete actions
- rewards
- episode termination signals
These are used to collect rollout data for PPO training.

### 3. Build Actor and Critic Networks
The actor and critic networks are defined in:
```text
src/networks/mlp.py
```
The actor network maps a CartPole state to action logits:
```text
state -> actor -> action logits -> action distribution
```
The critic network maps a CartPole state to a scalar value estimate:
```text
state -> critic -> V(s)
```
The actor is used to sample actions.
The critic is used to estimate values for advantage computation.

### 4. Collect Rollout Data
Rollout collection is implemented in:
```text
src/algorithms/ppo.py
```
During rollout collection, the current policy interacts with the environment and stores the sampled data into a rollout buffer.

The rollout buffer is defined in:
```
src/buffers/rollout_buffer.py
```
It stores `states`, `actions`, `rewards`, `dones`, `old_log_probs`, `values`.

Here:

- `old_log_probs` are the log probabilities under the policy that collected the data.
- `values` are critic estimates used for GAE.
- `dones` indicate whether an episode terminates at a given step.

### 5. Compute GAE Advantages and Returns
GAE computation is implemented in:
```
src/algorithms/ppo.py
```
The advantages are used to update the actor, while the returns are used as value targets for the critic.

Before PPO optimization, the advantages are normalized in:
```
src/trainers/train_ppo.py
```
This helps stabilize policy gradient updates.
### 6. PPO Minibatch Update
The main PPO training loop is implemented in:
```
src/trainers/train_ppo.py
```
The trainer repeatedly performs:
```
collect rollout
-> compute GAE
-> normalize advantages
-> split rollout data into minibatches
-> evaluate current policy on old states and actions
-> compute PPO ratio
-> compute clipped actor loss
-> compute critic loss
-> compute entropy loss
-> update actor and critic
```
The action evaluation function is implemented in:
```
src/algorithms/ppo.py
```
It computes `new_log_probs`, `entropy`, `new_values`.
These values are used to compute the PPO losses.

### 7. Losses Used in Training
Inside `src/trainers/train_ppo.py`, the total loss is composed of three parts:
```
total_loss = actor_loss + vf_coef * critic_loss + ent_coef * entropy_loss
```
where:
- `actor_loss` is the clipped PPO policy loss
- `critic_loss` is the value function regression loss
- `entropy_loss` encourages exploration

The PPO ratio is computed as:
```
ratio = exp(new_log_probs - old_log_probs)
```
The clipped surrogate objective prevents the new policy from moving too far away from the old policy that collected the rollout data.
### 8. Logging and Visualization
During training, the following metrics are recorded:
```
rewards_history
actor_loss_history
critic_loss_history
entropy_history
approx_kl_history
clip_fraction_history
```
The plotting function is implemented in:
```
src/utils/plot.py
```
The generated figure is saved as:
```
training_curves.png
```
The full metric history is saved as:
```
history.json
```
The run metadata is saved as:
```
metadata.json
```
The experiment configuration is saved as:
```
config.yaml
```

## Training Curves and Result Analysis

![training_curves](/rl_toy/results/ppo_cartpole/main_run/training_curves.png)

The training figure contains six curves: rollout reward, actor loss, critic loss, policy entropy, approximate KL divergence, and clip fraction.

### 1. Rollout Reward

Rollout reward measures the actual performance of the PPO agent in the CartPole environment.

In this run, the reward increases from a low value at the beginning to around 300–500 in later updates, and sometimes reaches the maximum reward of 500. This indicates that the agent successfully learns a balancing strategy.

The reward curve is not strictly monotonic because PPO is still updating a stochastic policy. Even after the policy improves, rollout performance may fluctuate due to exploration, sampling randomness, and policy updates.

### 2. Actor Loss

Actor loss corresponds to the PPO clipped policy objective.

The actor loss fluctuates around zero. This is normal because PPO uses normalized advantages and a clipped surrogate objective. The absolute value of actor loss is less important than whether the reward improves and whether KL / clip fraction remain within a reasonable range.

Large actor loss spikes may indicate unstable policy updates, but in this run the actor loss remains bounded while reward improves, suggesting that the policy update is generally stable.

### 3. Critic Loss

Critic loss measures how well the value network predicts the return target.

The critic loss fluctuates much more strongly than actor loss. This is expected because the value target changes as the policy improves and the collected trajectories become different. In CartPole, when episode lengths increase quickly, the value targets also become larger, which can cause temporary critic loss spikes.

A fluctuating critic loss does not necessarily mean PPO fails. It should be interpreted together with reward. In this run, reward keeps improving, so the critic is still useful for advantage estimation even though its regression loss is noisy.

### 4. Policy Entropy

Policy entropy measures how uncertain the action distribution is.

At the beginning, entropy is high because the policy is close to random. As training progresses, entropy gradually decreases, meaning the policy becomes more confident and more deterministic.

In this run, entropy decreases from around 0.69 to around 0.55. This is consistent with successful learning: the agent explores more in the beginning and gradually commits to a better balancing strategy.

### 5. Approximate KL

Approximate KL measures how much the new policy differs from the old policy that collected the rollout data.

```
approx_kl = (batch_old_log_probs-new_log_probs).mean().item()
```

In PPO, KL is important because the algorithm should improve the policy without changing it too aggressively. If KL becomes too large, the policy update may be unstable. If KL is always extremely small, learning may be too slow.

In this run, approximate KL mostly stays in a moderate range with occasional spikes. This suggests that PPO is updating the policy meaningfully while the clipped objective still prevents destructive policy shifts.

### 6. Clip Fraction

Clip fraction measures the proportion of samples whose policy ratio is clipped.

If clip fraction is close to zero all the time, PPO clipping is rarely activated, which may mean updates are too small. If clip fraction is too high, too many samples are being clipped, which may indicate overly aggressive updates.

In this run, clip fraction is usually low but occasionally spikes. This means the clipping mechanism is actively constraining some policy updates, especially when the new policy moves too far from the old policy on certain minibatches.

## Relation Between Curves and Hyperparameters

The training curves are strongly affected by PPO hyperparameters.

### Learning Rate

The actor learning rate controls how fast the policy changes.

- If actor learning rate is too large, approximate KL and clip fraction may spike frequently, and reward may become unstable.
- If actor learning rate is too small, reward may increase slowly and actor loss may stay close to zero with little improvement.

The critic learning rate controls how fast the value network fits return targets.

- If critic learning rate is too large, critic loss may oscillate strongly.
- If critic learning rate is too small, value estimation may lag behind policy improvement, making advantage estimation less reliable.

### Clip Epsilon

`clip_eps` controls how much the new policy is allowed to deviate from the old policy.

- Larger `clip_eps` allows larger policy updates, which may speed up learning but increase instability.
- Smaller `clip_eps` makes updates more conservative, which may improve stability but slow down learning.

If clip fraction is frequently high, reducing learning rate or reducing `clip_eps` may help.  
If clip fraction is almost always zero and reward improves slowly, increasing learning rate or `clip_eps` may help.

### Entropy Coefficient

`ent_coef` controls the strength of entropy regularization.

- Larger entropy coefficient encourages more exploration and keeps entropy higher.
- Smaller entropy coefficient allows the policy to become deterministic faster.

If entropy drops too quickly and reward gets stuck, increasing `ent_coef` may help.  
If the policy keeps exploring too much and reward does not stabilize, decreasing `ent_coef` may help.

### Rollout Steps

`rollout_steps` controls how much on-policy data is collected before each PPO update.

- Larger rollout batches give more stable advantage estimates but slower update frequency.
- Smaller rollout batches update more frequently but may introduce more variance.

### PPO Epochs and Minibatch Size

`ppo_epochs` controls how many times the same rollout batch is reused.

- More epochs improve data reuse but may make the new policy drift too far from the old policy.
- Fewer epochs are more conservative but may underuse collected data.

If KL and clip fraction are too high, reducing `ppo_epochs` can help.  
If learning is too slow and KL is very small, increasing `ppo_epochs` may help.t

## Hyperparameter Comparison

The current figure corresponds to the following setting:
```
actor_lr = 3e-4
critic_lr = 5e-4
clip_eps = 0.2
```
This setting successfully solves CartPole, as the rollout reward reaches the maximum reward of 500 in later training.

In future experiments, I will compare different hyperparameter settings, such as:

```
1. Smaller actor learning rate
2. Larger actor learning rate
3. Smaller clip_eps
4. Larger entropy coefficient
5. Different rollout_steps
```

The comparison will focus on:

- reward improvement speed
- final reward stability
- entropy decay speed
- critic loss stability
- approximate KL magnitude
- clip fraction frequency

Overall, this run shows a successful PPO training process on CartPole. The reward improves substantially and reaches the maximum value in later updates. Entropy decreases as the policy becomes more confident. KL and clip fraction remain active but not explosively large, suggesting that PPO clipping is helping constrain policy updates. The critic loss is noisy, but this does not prevent the policy from learning a successful control strategy.