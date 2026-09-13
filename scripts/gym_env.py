import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
from gymnasium.envs.registration import register

import numpy as np
import mujoco
import time

PATH_TO_MJCF = "/home/shourya/rl/urdf/arduinobot.xml"


class ArduinoBotEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, render_mode=None):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(PATH_TO_MJCF)
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.viewer = None

        self.arm_joint_names = ["joint_1", "joint_2", "joint_3"]
        self.ee_site_id = self.model.site("ee_site").id

        self.joint_ranges = []
        for name in self.arm_joint_names:
            jid = self.model.joint(name).id
            if self.model.jnt_limited[jid]:
                self.joint_ranges.append(tuple(self.model.jnt_range[jid]))
            else:
                self.joint_ranges.append((-np.pi, np.pi))

        low = np.array([r[0] for r in self.joint_ranges], dtype=np.float32)
        high = np.array([r[1] for r in self.joint_ranges], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        obs_dim = 3 + 3 + 3 + 3 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.target_pos = np.zeros(3, dtype=np.float32)

        # Task parameters
        self.success_threshold = 0.15       # meters
        self.velocity_threshold = 0.2      # m/s, for stability check

        # Load reachable workspace point cloud
        self.reachable_points = np.load("/home/shourya/ee_reachable_points.npy")

        # Scratch buffer reused every step for mj_objectVelocity
        self._ee_vel6 = np.zeros(6)

    def _get_obs(self):
        qpos = np.array([self.data.qpos[self.model.joint(name).qposadr]
                          for name in self.arm_joint_names]).flatten()
        qvel = np.array([self.data.qvel[self.model.joint(name).dofadr]
                          for name in self.arm_joint_names]).flatten()

        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        target_pos = self.target_pos.copy()
        error = target_pos - ee_pos

        obs = np.concatenate([qpos, qvel, ee_pos, target_pos, error]).astype(np.float32)
        return obs

    def _get_ee_speed(self):
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_SITE,
            self.ee_site_id,
            self._ee_vel6,
            0,
        )
        linear_vel = self._ee_vel6[3:6]
        return np.linalg.norm(linear_vel)

    def _get_reward(self):
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        distance = np.linalg.norm(self.target_pos - ee_pos)
        ee_speed = self._get_ee_speed()

        # Dense shaping: penalise distance and add a small step urgency penalty
        reward = -distance - 0.01

        # Small penalty for high-speed "flying through" the target region
        if distance < self.success_threshold:
            reward -= 0.1 * ee_speed

        # Success bonus (Increased from 10.0 to 100.0 to offset distance penalties)
        success = (distance < self.success_threshold) and (ee_speed < self.velocity_threshold)
        if success:
            reward += 100.0

        return reward, distance, ee_speed, success

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Sample a random reachable target with small jitter
        idx = self.np_random.integers(0, len(self.reachable_points))
        base_point = self.reachable_points[idx]
        jitter = self.np_random.uniform(-0.02, 0.02, size=3)
        self.target_pos = (base_point + jitter).astype(np.float32)

        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        for i, name in enumerate(self.arm_joint_names):
            actuator_id = self.model.actuator(f"servo_{name}").id
            self.data.ctrl[actuator_id] = action[i]

        for _ in range(5): # Run 5 physics steps per action step
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, distance, ee_speed, success = self._get_reward()

        # Let the Gymnasium TimeLimit wrapper manage truncation instead of manual counting
        terminated = bool(success)
        truncated = False

        info = {"is_success": success, "distance": distance, "ee_speed": ee_speed}

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# --- REGISTRATION AND RUNTIME ---
# Register the environment properly so Gymnasium wraps it with a TimeLimit
register(
    id="ArduinoBot-v0",
    entry_point="__main__:ArduinoBotEnv",
    max_episode_steps=1000, # Reduced to 1000 (5000 is exceptionally long for reaching tasks)
)

if __name__ == "__main__":
    # Create the wrapped environment instance
    env = gym.make("ArduinoBot-v0", render_mode="human")
    
    print("Action space low:", env.action_space.low)
    print("Action space high:", env.action_space.high)
    obs, info = env.reset()
    print("Initial obs:", obs)

    try:
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            time.sleep(0.02)
            if terminated or truncated:
                print(f"Episode ended. Reason: {'Success' if terminated else 'Timeout'}. Resetting.")
                obs, info = env.reset()
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        env.close()
