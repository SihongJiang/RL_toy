import gymnasium as gym

def make_env(env_name="CartPole-v1", render_mode=None):
    env = gym.make(env_name, render_mode=render_mode)
    return env