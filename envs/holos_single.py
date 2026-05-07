import numpy as np
import gymnasium as gym

from .holos_multi import HolosMulti


class HolosSingle(gym.Env):
    def __init__(self, profile, episode_length, run_path=None,
                 train_mode=True, noise=0.0, debug=False,
                 valid_maskings=(0,)):
        self.profile = profile
        self.multi_env = HolosMulti(profile, episode_length, run_path, train_mode, noise, debug, valid_maskings)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "drum_angle": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "power": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "last_power": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "next_desired_power": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        })
        self.multi_env.reset()

    def reset(self, seed=None, options=None):
        obs, info = self.multi_env.reset(seed=seed, options=options)
        self.time = self.multi_env.time
        observation = obs.copy()
        observation.pop('drum_angles', None)
        observation["drum_angle"] = np.array([np.mean(obs["drum_angles"])])  # treat as a single drum angle
        return observation, info

    def step(self, action):
        unwrapped_action = action.item()
        action = np.array([unwrapped_action] * 8)
        obs, reward, terminated, truncated, info = self.multi_env.step(action)
        self.time = self.multi_env.time
        observation = obs.copy()
        observation.pop('drum_angles', None)
        observation["drum_angle"] = np.array([np.mean(obs["drum_angles"])])  # treat as a single drum angle
        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        self.multi_env.render(mode=mode)
