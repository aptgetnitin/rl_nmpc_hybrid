"""Single source of truth for HolosGen microreactor physical constants.

Shared by `envs.HolosPK` (the true plant) and `nmpc.ReactorModel` (the
controller's internal model). Both used to redeclare these values, with
"MUST stay in sync" comments — keeping the numbers in one place removes
that footgun.

Parameter values mostly follow Choi 2020 (Table 2) for the Holos-Quad
microreactor.
"""
import numpy as np

# ---- Neutronics: prompt + 6 delayed-neutron groups ------------------------
neutron_lifetime = 1.68e-3            # Lambda, s
beta = 0.004801                       # total delayed-neutron fraction
betas = np.array([                    # per-group delayed-neutron fractions
    1.42481E-04, 9.24281E-04, 7.79956E-04,
    2.06583E-03, 6.71175E-04, 2.17806E-04,
])
betas.setflags(write=False)
lambdas = np.array([                  # per-group precursor decay constants, s^-1
    1.272E-02, 3.174E-02, 1.160E-01,
    3.110E-01, 1.400E+00, 3.870E+00,
])
lambdas.setflags(write=False)
Sigma_f = 0.1117                      # macroscopic fission xsec, m^-1 (Choi 2020 Fig.15c)
therm_n_vel = 2.19e3                  # thermal neutron velocity, m/s (~0.025 eV)
n_0 = 2.25e13                         # steady-state neutron number density, m^-3
P_r = 22e6                            # rated thermal power, W

# ---- Xenon-iodine poisoning (slow, hours timescale) -----------------------
sigma_Xe = 2.65e-22                   # microscopic Xe-135 absorption xsec, m^2
yield_I = 0.061                       # I-135 fission yield
yield_Xe = 0.002                      # direct Xe-135 fission yield
lambda_I = 2.87e-5                    # I-135 decay constant, s^-1 (T1/2 ~ 6.7 h)
lambda_Xe = 2.09e-5                   # Xe-135 decay constant, s^-1 (T1/2 ~ 9.2 h)

# ---- Thermal-hydraulics: 3-node lumped (fuel / moderator / coolant) -------
# Heat-flow chain: fission heat -> Tf -> Tm -> Tc -> flow out.
cp_f = 977                            # specific heat of fuel, J/(kg.K)
cp_m = 1697                           # specific heat of moderator, J/(kg.K)
cp_c = 5188.6                         # specific heat of coolant, J/(kg.K)
M_f = 2002                            # mass of fuel, kg
M_m = 11573                           # mass of moderator, kg
M_c = 500                             # mass of coolant (in-core inventory), kg
K_fm = 1.17e6                         # fuel <-> moderator thermal conductance, W/K
K_mc = 2.16e5                         # moderator <-> coolant thermal conductance, W/K
M_dot = 17.5                          # coolant mass flow rate, kg/s
heat_f = 0.96                         # fraction of fission heat in fuel ('q' in Choi 2020)

# ---- Steady-state temperatures (re-derived after the dTf/dt fix) ----------
# With n_r = 1 and the corrected fuel equation, dTf=dTm=dTc=0 give:
#   Tc0 = T_in + P_r / (2 * M_dot * cp_c)
#   Tm0 = Tc0 + P_r / K_mc
#   Tf0 = Tm0 + heat_f * P_r / K_fm
Tf0 = 1036.5178                       # K, fuel
Tm0 = 1018.4666                       # K, moderator
Tc0 = 916.6147                        # K, coolant (lumped, in-core)
T_in = 795.47                         # K, coolant inlet (boundary condition)
T_out = 1037.7594                     # K, implied outlet = 2*Tc0 - T_in (informational)

# ---- Reactivity feedback coefficients -------------------------------------
alpha_f = -2.875e-5                   # fuel (Doppler) reactivity coeff, 1/K (negative = stable)
alpha_m = -3.696e-5                   # moderator reactivity coeff, 1/K (negative = stable)
alpha_c = 0.0                         # coolant reactivity coeff, 1/K (unused)

# ---- Control drums (8 physical drums, each 0..180 deg) --------------------
u0 = 77.8                             # steady-state full-power drum angle, deg
rho_max = 0.00510                     # max reactivity per drum (510 pcm)


# ---- Derived constants (deterministic functions of the values above) ------
rho_ss = rho_max * (1 - np.cos(np.deg2rad(u0))) / 2
assert rho_ss < rho_max, 'steady state reactivity exceeds max reactivity'

I0 = yield_I * Sigma_f * therm_n_vel * n_0 / lambda_I
Xe0 = (
    (yield_Xe * Sigma_f * therm_n_vel * n_0 + lambda_I * I0)
    / (lambda_Xe + sigma_Xe * therm_n_vel * n_0)
)
