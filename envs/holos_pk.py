import numpy as np
from scipy.interpolate import interp1d

from . import holos_constants as _c


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

    All physical constants are sourced from `envs.holos_constants` so the
    plant and the NMPC controller's internal model can't drift apart.
    """

    # Neutronics
    neutron_lifetime = _c.neutron_lifetime
    beta = _c.beta
    betas = _c.betas
    lambdas = _c.lambdas
    Sigma_f = _c.Sigma_f
    therm_n_vel = _c.therm_n_vel
    n_0 = _c.n_0
    P_r = _c.P_r

    # Xenon-iodine
    sigma_Xe = _c.sigma_Xe
    yield_I = _c.yield_I
    yield_Xe = _c.yield_Xe
    lambda_I = _c.lambda_I
    lambda_Xe = _c.lambda_Xe

    # Thermal-hydraulics
    cp_f = _c.cp_f
    cp_m = _c.cp_m
    cp_c = _c.cp_c
    M_f = _c.M_f
    M_m = _c.M_m
    M_c = _c.M_c
    K_fm = _c.K_fm
    K_mc = _c.K_mc
    M_dot = _c.M_dot
    heat_f = _c.heat_f

    # Steady-state temperatures
    Tf0 = _c.Tf0
    Tm0 = _c.Tm0
    Tc0 = _c.Tc0
    T_in = _c.T_in
    T_out = _c.T_out

    # Reactivity feedback
    alpha_f = _c.alpha_f
    alpha_m = _c.alpha_m
    alpha_c = _c.alpha_c

    # Drums
    u0 = _c.u0
    rho_max = _c.rho_max

    def __init__(self):
        self.rho_ss = _c.rho_ss
        self.I0 = _c.I0
        self.Xe0 = _c.Xe0

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
        # in `holos_constants` were re-derived to be consistent with the
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
