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