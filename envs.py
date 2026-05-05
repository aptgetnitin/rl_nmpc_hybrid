import time
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import gymnasium as gym
from pettingzoo import ParallelEnv


class HolosPK:
    """Point-kinetics + thermal-hydraulics + xenon model of the HolosGen
    microreactor.

    The reactor is described by 12 coupled ODEs grouped into three physical
    subsystems on three different timescales:

        1. Neutron population n_r and 6 delayed-neutron precursor groups c_i
           (milliseconds-to-seconds dynamics; see `reactor_dae`).
        2. Three lumped temperatures Tf, Tm, Tc for fuel, moderator, coolant
           (seconds dynamics, energy chain Fission -> Fuel -> Moderator ->
           Coolant -> flow out).
        3. Iodine I and Xenon Xe number densities (hours dynamics; the famous
           xenon transient that makes large power swings hard to control).

    Operational input is the 8 control-drum angles (degrees, 0..180). Reactor
    power is the dimensionless n_r (1.0 == rated 22 MW). See `calc_reactivity`
    for how drums and feedbacks combine into the reactivity rho.

    Parameter values mostly follow Choi 2020 (Table 2) for the Holos-Quad
    microreactor.
    """

    ###########################################################################
    # Neutronics: prompt + 6 delayed-neutron groups
    ###########################################################################
    neutron_lifetime = 1.68e-3        # Lambda, s
    beta = 0.004801                   # total delayed-neutron fraction
    betas = np.array([                # per-group delayed-neutron fractions
        1.42481E-04, 9.24281E-04, 7.79956E-04,
        2.06583E-03, 6.71175E-04, 2.17806E-04,
    ])
    lambdas = np.array([              # per-group precursor decay constants, s^-1
        1.272E-02, 3.174E-02, 1.160E-01,
        3.110E-01, 1.400E+00, 3.870E+00,
    ])
    Sigma_f = 0.1117                  # macroscopic fission xsec, m^-1 (Choi 2020 Fig.15c)
    therm_n_vel = 2.19e3              # thermal neutron velocity, m/s (~0.025 eV)
    n_0 = 2.25e13                     # steady-state neutron number density, m^-3
    P_r = 22e6                        # rated thermal power, W

    ###########################################################################
    # Xenon-iodine poisoning (slow, hours timescale)
    ###########################################################################
    sigma_Xe = 2.65e-22               # microscopic Xe-135 absorption xsec, m^2
    yield_I = 0.061                   # I-135 fission yield
    yield_Xe = 0.002                  # direct Xe-135 fission yield
    lambda_I = 2.87e-5                # I-135 decay constant, s^-1 (T1/2 ~ 6.7 h)
    lambda_Xe = 2.09e-5               # Xe-135 decay constant, s^-1 (T1/2 ~ 9.2 h)

    ###########################################################################
    # Thermal-hydraulics: 3-node lumped model (fuel / moderator / coolant)
    # Heat flow chain: fission heat -> Tf -> Tm -> Tc -> flow out
    ###########################################################################
    cp_f = 977                        # specific heat of fuel, J/(kg.K)
    cp_m = 1697                       # specific heat of moderator, J/(kg.K)
    cp_c = 5188.6                     # specific heat of coolant, J/(kg.K)
    M_f = 2002                        # mass of fuel, kg
    M_m = 11573                       # mass of moderator, kg
    M_c = 500                         # mass of coolant (in-core inventory), kg
    K_fm = 1.17e6                     # fuel <-> moderator thermal conductance, W/K
    K_mc = 2.16e5                     # moderator <-> coolant thermal conductance, W/K
    M_dot = 17.5                      # coolant mass flow rate, kg/s
    heat_f = 0.96                     # fraction of fission heat deposited in fuel ('q' in Choi 2020)

    # Steady-state temperatures, RE-DERIVED after fixing the dTf/dt bug below
    # (previously this term used (Tf - Tc), which broke energy conservation
    # between the fuel and moderator nodes; see comment in reactor_dae).
    # With n_r = 1 and the corrected equation, dTf=dTm=dTc=0 give:
    #   Tc0 = T_in + P_r / (2 * M_dot * cp_c)
    #   Tm0 = Tc0 + P_r / K_mc
    #   Tf0 = Tm0 + heat_f * P_r / K_fm
    Tf0 = 1036.5178                   # K, fuel
    Tm0 = 1018.4666                   # K, moderator
    Tc0 = 916.6147                    # K, coolant (lumped, in-core)
    T_in = 795.47                     # K, coolant inlet (boundary condition)
    T_out = 1037.7594                 # K, implied outlet = 2*Tc0 - T_in (informational)

    ###########################################################################
    # Reactivity feedbacks (material properties; reference the steady state above)
    ###########################################################################
    alpha_f = -2.875e-5               # fuel (Doppler) reactivity coeff, 1/K (negative = stable)
    alpha_m = -3.696e-5               # moderator reactivity coeff, 1/K (negative = stable)
    alpha_c = 0.0                     # coolant reactivity coeff, 1/K (unused)

    ###########################################################################
    # Control drums (8 physical drums, each 0..180 deg)
    ###########################################################################
    u0 = 77.8                         # steady-state full-power drum angle, deg
    rho_max = 0.00510                 # max reactivity per drum (510 pcm)

    def __init__(self):
        # calculate steady state conditions and drum reactivity
        self.rho_ss = self.rho_max * (1 - np.cos(np.deg2rad(self.u0))) / 2
        assert self.rho_ss < self.rho_max, 'steady state reactivity exceeds max reactivity'
        self.I0 = self.yield_I * self.Sigma_f * self.therm_n_vel * self.n_0 / self.lambda_I
        self.Xe0 = ((self.yield_Xe * self.Sigma_f * self.therm_n_vel * self.n_0
                     + self.lambda_I * self.I0)
                    / (self.lambda_Xe
                       + self.sigma_Xe * self.therm_n_vel * self.n_0))

    def get_initial_conditions(self):
        n_r = 1
        c1, c2, c3, c4, c5, c6 = [n_r] * 6
        Tf = self.Tf0
        Tm = self.Tm0
        Tc = self.Tc0
        Xe = self.Xe0
        I = self.I0

        return [n_r, c1, c2, c3, c4, c5, c6, Tf, Tm, Tc, Xe, I]

    def calc_reactivity(self, y, drum_angles):
        """Total reactivity rho = sum of four contributions:

            1. Drum reactivity:    operator-controlled, sin^2(u/2) per drum,
                                   referenced so that all 8 drums at u0 give 0.
            2. Doppler feedback:   alpha_f * (Tf - Tf0). Negative -> stabilizing.
            3. Moderator feedback: alpha_m * (Tm - Tm0). Negative -> stabilizing.
            4. Xenon poison:       -sigma_Xe * (Xe - Xe0) / Sigma_f. Note this
                                   uses Sigma_f (fission) as an approximation
                                   to Sigma_a (absorption); near criticality
                                   Sigma_a ~ Sigma_f, so this is the standard
                                   simplification used in Choi 2020.
        """
        _, _, _, _, _, _, _, Tf, Tm, _, Xe, _ = y

        drum_reactivity = np.sum(self.rho_max * (1 - np.cos(np.deg2rad(drum_angles))) / 2 - self.rho_ss)
        assert drum_reactivity < self.rho_max, 'drum reactivity exceeds max reactivity'
        rho = (drum_reactivity
               + self.alpha_f * (Tf - self.Tf0)
               + self.alpha_m * (Tm - self.Tm0)
               - self.sigma_Xe * (Xe - self.Xe0) / self.Sigma_f)

        return rho

    def drum_forcing(self, drum_angles, drum_action, time = 1):
        """Create a drum angle forcer for intermediate timesteps during a solve_ivp"""
        drum_forcers = []
        for i, drum_angle in enumerate(drum_angles):
            new_angle = np.clip(drum_angle + drum_action[i], 0, 180).item()  # can't go beyond limits
            drum_forcers.append(interp1d([0, time], [drum_angle, new_angle]))

        assert len(drum_forcers) == len(drum_angles)
        return drum_forcers

    def reactor_dae(self, t, y, d1, d2, d3, d4, d5, d6, d7, d8):
        """Right-hand side of the 12-state reactor ODE.

        State y = [n_r, c1..c6, Tf, Tm, Tc, Xe, I]:
            n_r       -- normalized neutron population (1.0 = rated power)
            c1..c6    -- normalized delayed-precursor concentrations
            Tf, Tm, Tc -- fuel / moderator / coolant temperatures, K
            Xe, I     -- xenon and iodine number densities

        Inputs d1..d8 are time-functions (one per drum); the integrator can
        evaluate them mid-step, which lets us linearly ramp drum angles within
        a 1 s control interval rather than apply an instantaneous jump.

        See `calc_reactivity` for how drum angles, temperatures, and xenon
        combine into the master rho knob below.
        """
        n_r, c1, c2, c3, c4, c5, c6, Tf, Tm, Tc, Xe, I = y
        drum_angles = np.array([d1(t), d2(t), d3(t), d4(t), d5(t), d6(t), d7(t), d8(t)])
        rho = self.calc_reactivity(y, drum_angles)
        precursor_concentrations = np.array([c1, c2, c3, c4, c5, c6])

        # ---- Point kinetics: neutron population + 6 delayed-precursor groups ----
        # Normalized form: c_i is c_i_actual / c_i_steady_state, so
        # dc_i/dt = lambda_i * (n_r - c_i) and the n_r equation uses beta_i (not
        # beta_i / Lambda) inside the sum.
        d_n_r = (((rho - self.beta) * n_r
                  + np.sum(self.betas * precursor_concentrations))
                 / self.neutron_lifetime)
        d_c1, d_c2, d_c3, d_c4, d_c5, d_c6 = (self.lambdas * n_r
                                              - self.lambdas * precursor_concentrations)

        # ---- Thermal-hydraulics: fission heat -> Fuel -> Moderator -> Coolant -> flow out ----
        # FIX: previously the second term of d_Tf was K_fm*(Tf - Tc), which
        # broke energy conservation -- heat *leaving* the fuel did not match
        # heat *entering* the moderator (which correctly uses K_fm*(Tf - Tm)
        # below). The fuel sits inside the moderator, not the coolant, so the
        # conduction must be through Tm. The steady-state initial temperatures
        # at the top of this class were re-derived to be consistent with the
        # corrected equation below.
        d_Tf = ((self.heat_f * self.P_r * n_r
                 - self.K_fm * (Tf - Tm))
                / (self.M_f * self.cp_f))

        d_Tm = (((1 - self.heat_f) * self.P_r * n_r
                 + self.K_fm * (Tf - Tm)
                 - self.K_mc * (Tm - Tc))
                / (self.M_m * self.cp_m))

        # Coolant: heat in from moderator, heat out by flow. The factor of 2
        # comes from the lumped-coolant approximation T_out = 2*Tc - T_in,
        # so the carried-away enthalpy is M_dot*cp_c*(T_out - T_in)
        # = 2*M_dot*cp_c*(Tc - T_in).
        d_Tc = ((self.K_mc * (Tm - Tc)
                 - 2 * self.M_dot * self.cp_c * (Tc - self.T_in))
                / (self.M_c * self.cp_c))

        # ---- Xenon-iodine poisoning (slow, hours timescale) ----
        # n_rate_density = phi (thermal flux) since n_r is normalized by n_0.
        # I-135 builds from fission and decays to Xe-135. Xe-135 builds from
        # both direct fission and I-135 decay; it disappears either by decay or
        # by absorbing a neutron (which is also what makes it a strong poison).
        n_rate_density = self.therm_n_vel * self.n_0 * n_r
        d_I = (self.yield_I * self.Sigma_f * n_rate_density
               - self.lambda_I * I)
        d_Xe = (self.yield_Xe * self.Sigma_f * n_rate_density
                + self.lambda_I * I
                - self.lambda_Xe * Xe
                - self.sigma_Xe * Xe * n_rate_density)

        return [d_n_r, d_c1, d_c2, d_c3, d_c4, d_c5, d_c6, d_Tf, d_Tm, d_Tc, d_Xe, d_I]


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
