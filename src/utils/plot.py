import os
import matplotlib.pyplot as plt


def plot_training_curves(history, output_dir, filename="training_curves.png"):
    rewards_history = history["rewards_history"]
    actor_loss_history = history["actor_loss_history"]
    critic_loss_history = history["critic_loss_history"]
    entropy_history = history["entropy_history"]
    approx_kl_history = history["approx_kl_history"]
    clip_fraction_history = history["clip_fraction_history"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    axes[0].plot(rewards_history)
    axes[0].set_title("Rollout Reward")
    axes[0].set_xlabel("Update")
    axes[0].set_ylabel("Reward")
    axes[0].grid(True)

    axes[1].plot(actor_loss_history)
    axes[1].set_title("Actor Loss")
    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True)

    axes[2].plot(critic_loss_history)
    axes[2].set_title("Critic Loss")
    axes[2].set_xlabel("Training Step")
    axes[2].set_ylabel("Loss")
    axes[2].grid(True)

    axes[3].plot(entropy_history)
    axes[3].set_title("Policy Entropy")
    axes[3].set_xlabel("Training Step")
    axes[3].set_ylabel("Entropy")
    axes[3].grid(True)

    axes[4].plot(approx_kl_history)
    axes[4].set_title("Approx KL")
    axes[4].set_xlabel("Training Step")
    axes[4].set_ylabel("KL")
    axes[4].grid(True)

    axes[5].plot(clip_fraction_history)
    axes[5].set_title("Clip Fraction")
    axes[5].set_xlabel("Training Step")
    axes[5].set_ylabel("Fraction")
    axes[5].grid(True)

    plt.tight_layout()

    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Figure saved to: {save_path}")
    return save_path


def plot_reinforce_training_curves(
    history,
    output_dir,
    filename="reinforce_training_curves.png"
):
    import os
    import matplotlib.pyplot as plt

    rewards_history = history["rewards_history"]
    rolling_reward_mean_history = history["rolling_reward_mean_history"]
    rolling_return_variance = history["rolling_return_variance"]
    raw_returns_variance_history = history["raw_returns_variance_history"]
    policy_loss_variance_history = history["policy_loss_variance_history"]
    gradient_norm_history = history["gradient_norm_history"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    axes[0, 0].plot(rewards_history)
    axes[0, 0].set_title("Episode Reward")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].grid(True)

    axes[0, 1].plot(rolling_reward_mean_history)
    axes[0, 1].set_title("Rolling Mean Reward (window=50)")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Mean Reward")
    axes[0, 1].grid(True)

    axes[1, 0].plot(rolling_return_variance)
    axes[1, 0].set_title("Rolling Reward Variance (window=50)")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Variance")
    axes[1, 0].grid(True)

    axes[1, 1].plot(raw_returns_variance_history)
    axes[1, 1].set_title("Monte Carlo Return Variance")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Variance")
    axes[1, 1].grid(True)

    axes[2, 0].plot(policy_loss_variance_history)
    axes[2, 0].set_title("Policy Loss Variance")
    axes[2, 0].set_xlabel("Episode")
    axes[2, 0].set_ylabel("Variance")
    axes[2, 0].grid(True)

    axes[2, 1].plot(gradient_norm_history)
    axes[2, 1].set_title("Gradient Norm")
    axes[2, 1].set_xlabel("Episode")
    axes[2, 1].set_ylabel("L2 Norm")
    axes[2, 1].grid(True)

    plt.tight_layout()

    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[INFO] REINFORCE figure saved to: {save_path}")

    return save_path