import sys
import os
import time

sys.path.append(os.path.dirname(__file__))  # so it can find gym_env.py in the same folder

from gym_env import ArduinoBotEnv

env = ArduinoBotEnv(render_mode="human")
obs, info = env.reset()

extremes = [
    [1.5708, 1.5708, 1.5708],
    [-1.5708, -1.5708, -1.5708],
    [1.5708, -1.5708, 1.5708],
    [-1.5708, 1.5708, -1.5708],
]

try:
    for target in extremes:
        for _ in range(200):
            obs, reward, terminated, truncated, info = env.step(target)
            env.render()
            time.sleep(0.01)
except KeyboardInterrupt:
    print("Stopped by user.")
finally:
    env.close()
