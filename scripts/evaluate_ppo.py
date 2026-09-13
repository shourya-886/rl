"""
Evaluate a trained PPO policy on the ArduinoBot reach task.

Usage:
    python3 evaluate_ppo.py                     # headless, N_EPISODES stats
    python3 evaluate_ppo.py --render            # watch it live in the MuJoCo viewer
    python3 evaluate_ppo.py --model path/to.zip # evaluate a specific checkpoint
"""

import os
import sys
import argparse
import time

sys.path.append(os.path.dirname(__file__))

import numpy as np
from gym_env import ArduinoBotEnv
from stable_baselines3 import PPO


DEFAULT_MODEL_PATH = "/home/shourya/rl/models/ppo_arduinobot/best_model.zip"
N_EPISODES = 50


def evaluate(model_path, render, n_episodes):
    render_mode = "human" if render else None
    env = ArduinoBotEnv(render_mode=render_mode)
    model = PPO.load(model_path)

    successes = 0
    episode_rewards = []
    episode_lengths = []
    final_distances = []

    try:
        for ep in range(n_episodes):
            obs, info = env.reset()
            done = False
            ep_reward = 0.0
            ep_len = 0
            last_distance = None

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                ep_len += 1
                last_distance = info["distance"]
                done = terminated or truncated

                if render:
                    env.render()
                    time.sleep(0.01)

            success = info["is_success"]
            successes += int(success)
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_len)
            final_distances.append(last_distance)

            status = "SUCCESS" if success else "FAILED "
            print(f"Episode {ep:3d} [{status}] "
                  f"reward={ep_reward:8.3f}  steps={ep_len:4d}  "
                  f"final_distance={last_distance:.4f}")

    except KeyboardInterrupt:
        print("Evaluation interrupted by user.")

    finally:
        env.close()

    n_done = len(episode_rewards)
    if n_done > 0:
        print("\n=== Evaluation Summary ===")
        print(f"Episodes run:       {n_done}")
        print(f"Success rate:       {successes}/{n_done} ({100.0 * successes / n_done:.1f}%)")
        print(f"Mean reward:        {np.mean(episode_rewards):.3f}")
        print(f"Mean episode length:{np.mean(episode_lengths):.1f}")
        print(f"Mean final distance:{np.mean(final_distances):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH,
                         help="Path to the trained model .zip file")
    parser.add_argument("--render", action="store_true",
                         help="Render episodes live in the MuJoCo viewer")
    parser.add_argument("--episodes", type=int, default=N_EPISODES,
                         help="Number of evaluation episodes to run")
    args = parser.parse_args()

    evaluate(args.model, args.render, args.episodes)