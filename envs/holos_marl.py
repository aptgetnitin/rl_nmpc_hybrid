import numpy as np
import gymnasium as gym
from pettingzoo import ParallelEnv

from .holos_multi import HolosMulti


class HolosMARL(ParallelEnv):
    metadata = {"render_modes": ["dataframe"], "name": "holos_marl_v0"}
    def __init__(self, profile, episode_length, run_path=None,
                 train_mode=True, noise=0.0, debug=False,
                 valid_maskings=(0,)):
        super().__init__()
        self.render_mode = "dataframe"
        self.gym_env = HolosMulti(profile, episode_length, run_path, train_mode, noise, debug, valid_maskings)
        self.agents = [f"agent_{i}" for i in range(8)]  # 8 control drums
        self.possible_agents = self.agents[:]

        # Each agent represents one out of the eight control drums
        self._action_spaces = {
            agent: gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
            for agent in self.agents
        }

        # Each agent gets the same observations
        self._observation_spaces = {
            agent: gym.spaces.Dict({
                "drum_angle": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "power": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "last_power": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "next_desired_power": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            })
            for agent in self.agents
        }

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        obs, info = self.gym_env.reset(seed=seed)

        observations = {agent: obs.copy() for agent in self.agents}
        for agent in self.agents:
            observations[agent].pop('drum_angles', None)
            index = int(agent.split("_")[-1])
            observations[agent]["drum_angle"] = np.array([obs["drum_angles"][index]])
        infos = {agent: info for agent in self.agents}

        return observations, infos

    def step(self, actions):
        # Combine actions from all agents
        action = np.array([actions[agent].item() for agent in self.agents])

        # Step the environment with combined action
        obs, reward, terminated, truncated, info = self.gym_env.step(action)

        # Distribute observations, rewards, and other info to all agents
        observations = {agent: obs.copy() for agent in self.agents}
        for agent in self.agents:
            observations[agent].pop('drum_angles', None)
            index = int(agent.split("_")[-1])
            observations[agent]["drum_angle"] = np.array([obs["drum_angles"][index]])
        rewards = {agent: (reward) for agent in self.agents}
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        infos = {agent: info for agent in self.agents}

        return observations, rewards, terminations, truncations, infos

    def render(self):
        return self.gym_env.render()

    def close(self):
        self.gym_env.close()
