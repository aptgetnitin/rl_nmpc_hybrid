import numpy as np
from scipy.interpolate import interp1d


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
