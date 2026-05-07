import time

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import gymnasium as gym

from .holos_pk import HolosPK


class HolosMulti(gym.Env):
    def __init__(self, profile, episode_length, run_path=None,
                 train_mode=True, noise=0.0, debug=False,
                 valid_maskings=(0,), symmetry_reward=False):
        self.profile = profile
        self.episode_length = episode_length
        self.run_path = run_path
        self.train_mode = train_mode
        self.noise = noise
        self.debug = debug
        self.valid_maskings = valid_maskings
        self.symmetry_reward = symmetry_reward

        self.pke = HolosPK()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "drum_angles": gym.spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32),
            "power": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "last_power": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "next_desired_power": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        })

        self.reset()

    def reset(self, seed=None, options=None):
        super(self.__class__, self).reset(seed=seed)
        current_desired_power = self.profile(0) / 100
        assert current_desired_power == 1, 'current code assumes start at full power steady state'
        self.time = 0
        self.drum_angles = np.array([77.8]*8)
        self.y = self.pke.get_initial_conditions()
        current_power, *_ = self.y
        assert current_power == 1, 'current code assumes start at full power steady state'

        num_masks = np.random.choice(self.valid_maskings)
        self.masks = np.ones(8)
        mask_indices = np.random.choice(8, size=num_masks, replace=False)
        self.masks[mask_indices] = 0
        assert np.sum(self.masks) == 8 - num_masks, 'error in mask assignment'

        next_desired_power = self.profile(self.time + 1)
        fuzz = np.random.normal(0, self.noise)
        fuzzed = current_power + fuzz
        self.history = [[self.time, *self.drum_angles, fuzzed, current_desired_power, *self.y]]
        observation = {
            "drum_angles": self.drum_angles / 180,  # convert to 0 to 1 box space
            "power": np.array([fuzzed]),
            "last_power": np.array([fuzzed]),
            "next_desired_power": np.array([next_desired_power / 100]),
        }

        return observation, {
            'latest': self.history[-1],
            'time': float(self.time),
            'desired_power': float(current_desired_power),
            'constraint_violation': False,
            'unsafe': False,
        }

    def gym2real_action(self, gym_action):
        """Convert from the -1 to 1 box space to -0.5 to 0.5"""
        assert type(gym_action) == np.ndarray, 'action must be a numpy array'
        real_action = gym_action / 2
        return real_action

    def real2gym_action(self, real_action):
        """Convert from the real -0.5 to 0.5 to the 0 1 continuous gym action space"""
        gym_action = real_action * 2
        assert type(gym_action) == np.ndarray, 'action must be a numpy array'
        return gym_action

    def step(self, action):
        if self.time >= self.episode_length:
            raise RuntimeError("Episode length exceeded")
        real_action = self.gym2real_action(action) * self.masks
        drum_forcers = self.pke.drum_forcing(self.drum_angles, real_action)
        sol = solve_ivp(self.pke.reactor_dae, [0, 1], self.y, args=drum_forcers)
        self.y = sol.y[:,-1]
        self.drum_angles += real_action
        self.drum_angles = np.clip(self.drum_angles, 0, 180)
        current_desired_power = self.profile(self.time) / 100
        self.time += 1

        current_power, *_ = self.y
        assert current_power >= 0 and current_power <= 2, 'power out of reasonable bounds'
        next_desired_power = self.profile(self.time + 1)
        fuzz = np.random.normal(0, self.noise)
        fuzzed = current_power + fuzz
        self.history.append([self.time, *self.drum_angles, fuzzed, current_desired_power, *self.y])
        assert len(self.history) == self.time + 1, 'history length mismatch'
        observation = {
            "drum_angles": self.drum_angles / 180,  # convert to 0-1 box space
            "power": np.array([fuzzed]),
            "last_power": np.array([self.history[-2][9]]),  # 9 is the measured power index
            "next_desired_power": np.array([next_desired_power / 100]),  # convert to 0-1 box space
        }

        desired_power = self.profile(self.time) / 100
        reward, terminated = self.calc_reward(current_power, desired_power)
        if current_power > 1.1:  # 110% power is way too much
            terminated = True
        if self.symmetry_reward:
            reward -= abs(np.max(action) - np.min(action))
        assert reward <= 2, 'max reward exceeded'
        truncated = False
        if self.time >= self.episode_length - 1:
            truncated = True
        diff = 100.0 * abs(float(current_power) - float(desired_power))
        constraint_violation = bool(diff > 5.0 or self.drum_angles.min() <= 0.0 or self.drum_angles.max() >= 180.0)
        unsafe = bool(float(current_power) > 1.1)
        info = {
            'latest': self.history[-1],
            'time': float(self.time),
            'desired_power': float(desired_power),
            'power_error': float(float(current_power) - float(desired_power)),
            'constraint_violation': constraint_violation,
            'unsafe': unsafe,
            'drum_min_deg': float(self.drum_angles.min()),
            'drum_max_deg': float(self.drum_angles.max()),
        }

        return observation, reward, terminated, truncated, info

    def calc_reward(self, current_power, desired_power):
        """Returns reward and whether the episode is terminated."""
        # First component: give reward to stay in the correct range
        diff = 100*abs(current_power - desired_power)
        assert diff <= 200, 'diff out of reasonable bounds'
        reward = 2 - diff

        # give a punish outside bounds if in train mode
        terminated = False
        if (self.train_mode and
            (diff > 5
            or self.drum_angles.min() <= 0
            or self.drum_angles.max() >= 180)):
            reward -= 100
            terminated = True

        return reward, terminated

    def render(self, mode='human'):
        run_history = np.array(self.history)
        column_names = ['time', 'drum_1', 'drum_2', 'drum_3', 'drum_4',
                        'drum_5', 'drum_6', 'drum_7', 'drum_8', 'measured_power',
                        'desired_power', 'actual_power', 'c1', 'c2', 'c3',
                        'c4', 'c5', 'c6', 'Tf', 'Tm', 'Tc', 'Xe', 'I']
        df = pd.DataFrame(run_history, columns=column_names)
        df['diff'] = (df['actual_power'] - df['desired_power']) * 100
        assert df['actual_power'][0] == 1, 'steady state initial power value should be 100'
        assert df['drum_1'][0] == 77.8, 'steady state initial drum angle should be 77.8'
        self.history = df

        if self.run_path is not None:
            assert self.run_path.is_dir(), 'run_path must be a valid directory'
            timestr = time.strftime("%Y%m%d-%H%M%S")
            save_path = self.run_path / f'run_history_{timestr}.csv'
            df.to_csv(save_path, index=False)
