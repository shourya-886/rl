"""
Train a PPO policy on the ArduinoBot reach task.

Usage:
    python3 train_ppo.py
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))

import gymnasium as gym  # Ensure we use the proper Gymnasium module
from gymnasium.envs.registration import register, registry
from gym_env import ArduinoBotEnv

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# ---- Config ----
TOTAL_TIMESTEPS = 1_000_000
LOG_DIR = "/home/shourya/rl/logs/ppo_arduinobot"
MODEL_SAVE_PATH = "/home/shourya/rl/models/ppo_arduinobot"
CHECKPOINT_FREQ = 50_000
EVAL_FREQ = 25_000
N_EVAL_EPISODES = 10

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# ---- Environment Registration ----
# This ensures that gym.make() applies the TimeLimit wrapper natively
ENV_ID = "ArduinoBot-v0"
if ENV_ID not in registry:
    register(
        id=ENV_ID,
        entry_point="gym_env:ArduinoBotEnv",
        max_episode_steps=1000,  # Limits episode length cleanly via Gymnasium
    )


def make_env():
    # Calling gym.make applies the TimeLimit wrapper automatically
    env = gym.make(ENV_ID, render_mode="human")
    env = Monitor(env)  # tracks episode reward/length, needed for eval callback logging
    return env


def main():
    train_env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=MODEL_SAVE_PATH,
        name_prefix="ppo_arduinobot_ckpt",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_SAVE_PATH,
        log_path=LOG_DIR,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        learning_rate=3e-4,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    final_path = os.path.join(MODEL_SAVE_PATH, "ppo_arduinobot_final")
    model.save(final_path)
    print(f"Training complete. Final model saved to {final_path}.zip")


if __name__ == "__main__":
    main()
