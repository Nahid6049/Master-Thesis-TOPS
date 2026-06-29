# -*- coding: utf-8 -*-
"""
PLL Sensitivity + Finish PLL–VSC Interaction
- Runs 3 cases: T_pll = 0.02, 0.1, 0.2
- Disturbances: Fault at B8 (2–2.1 s), Load increase at B6 (t>10 s)
- Outputs added:
    (1) PLL freq estimate vs system freq
    (2) PLL angle tracking vs voltage angle
    (3) Angle error (wrapped)
    (4) dq voltages (Vd, Vq)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import importlib

import tops.dynamic as dps
import tops.solvers as dps_sol

# ---------------------------------------------------------
# PATH FIX (same pattern you used)
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
base_work_dir = os.path.dirname(current_dir)
examples_dir = os.path.dirname(base_work_dir)
tops_root = os.path.dirname(examples_dir)

sys.path.append(base_work_dir)
sys.path.append(tops_root)

import examples.user_models.user_lib as user_lib


# ---------------------------------------------------------
# PLL CASES
# ---------------------------------------------------------
pll_cases = [0.02, 0.1, 0.2]
labels = ['Tpll = 0.02', 'Tpll = 0.1', 'Tpll = 0.2']

# No blue: green / orange / red
colors = ['green', 'orange', 'red']
linestyles = ['-', '--', '-.']
LW = 1  # thinner lines


# ---------------------------------------------------------
# Utility: wrap angle to [-pi, pi]
# ---------------------------------------------------------
def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------
# SIMULATION FUNCTION
# ---------------------------------------------------------
def run_simulation(T_pll_value, t_end=20.0, max_step=5e-3):

    import my_network as model_data
    importlib.reload(model_data)
    model = model_data.load()

    # Update T_pll in model (your VSC_SI table index [1][10] is T_pll)
    model['vsc']['VSC_SI'][1][10] = T_pll_value

    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_lib)
    ps.init_dyn_sim()

    sol = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0,
        ps.x_0.copy(),
        t_end,
        max_step=max_step
    )

    # Bus mapping (your reduced order)
    idx = {'B1': 0, 'B2': 1, 'B5': 2, 'B6': 3, 'B7': 4, 'B8': 5, 'B9': 6, 'B10': 7}
    iB6 = idx['B6']
    iB8 = idx['B8']

    res = defaultdict(list)
    t = 0.0

    while t < t_end:

        # -------------------------
        # Disturbances
        # -------------------------
        # Load change at B6 after 10s
        if t > 10:
            ps.y_bus_red_mod[iB6, iB6] = -0.1
        else:
            ps.y_bus_red_mod[iB6, iB6] = 0

        # Fault at B8 (2–2.1s)
        if 2 < t < 2.1:
            ps.y_bus_red_mod[iB8, iB8] = 1000
        else:
            ps.y_bus_red_mod[iB8, iB8] = 0

        # -------------------------
        # Step solver
        # -------------------------
        x = sol.y
        sol.step()
        t = sol.t
        v = sol.v

        for mdl in ps.dyn_mdls:
            mdl.reset_outputs()

        # -------------------------
        # Extract signals
        # -------------------------
        Xvsc = ps.vsc['VSC_SI'].local_view(x)

        # PLL angle (state)
        theta_pll = float(np.squeeze(Xvsc['angle']))

        # Voltage angle at B8 (actual angle PLL tries to track)
        theta_v = float(np.angle(v[iB8]))

        # PLL dq voltage components:
        # v_dq = v_t * exp(-j*theta_pll)
        v_t = v[iB8]
        v_dq = v_t * np.exp(-1j * theta_pll)
        Vd = float(np.real(v_dq))
        Vq = float(np.imag(v_dq))  # should match v_q method

        # PLL frequency estimate (Hz) from your model method
        f_pll = float(np.squeeze(ps.vsc['VSC_SI'].freq_est(x, v)))

        # VSC powers
        P_vsc = float(np.squeeze(ps.vsc['VSC_SI'].p_e(x, v) * ps.sys_data['s_n']))
        Q_vsc = float(np.squeeze(ps.vsc['VSC_SI'].q_e(x, v) * ps.sys_data['s_n']))

        # System frequency from generators (your previous approach)
        SpeedVector = ps.gen['GEN'].speed(x, v)
        f_sys = float(50 + 50 * np.mean(SpeedVector))

        # dq currents (states)
        i_d = float(np.squeeze(Xvsc['i_d']))
        i_q = float(np.squeeze(Xvsc['i_q']))

        # -------------------------
        # Store
        # -------------------------
        res['t'].append(t)

        res['theta_pll'].append(theta_pll)
        res['theta_v'].append(theta_v)

        res['Vd'].append(Vd)
        res['Vq'].append(Vq)

        res['f_pll'].append(f_pll)
        res['f_sys'].append(f_sys)

        res['i_d'].append(i_d)
        res['i_q'].append(i_q)

        res['P_vsc'].append(P_vsc)
        res['Q_vsc'].append(Q_vsc)

    return res


# ---------------------------------------------------------
# RUN ALL CASES
# ---------------------------------------------------------
results = []
for Tpll in pll_cases:
    print(f"Running simulation with T_pll = {Tpll}")
    results.append(run_simulation(Tpll))


# ---------------------------------------------------------
# PLOTTING HELPERS
# ---------------------------------------------------------
def plot_multi(title, y_key, y_label, xlim=None):
    plt.figure()
    for i in range(len(results)):
        t = np.array(results[i]['t'])
        y = np.array(results[i][y_key])
        plt.plot(
            t, y,
            color=colors[i],
            linestyle=linestyles[i],
            linewidth=LW,
            label=labels[i]
        )
    plt.xlabel('Time [s]')
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    if xlim is not None:
        plt.xlim(xlim)


# ---------------------------------------------------------
# (1) PLL phase error (Vq)  ✅
# ---------------------------------------------------------
plot_multi('PLL Sensitivity: Phase Error (Vq)', 'Vq', 'Vq (pu)')

# Optional zoom near fault
plot_multi('PLL Phase Error (Zoom near fault)', 'Vq', 'Vq (pu)', xlim=(1.9, 2.6))


# ---------------------------------------------------------
# (2) PLL frequency estimate vs System frequency  ✅
# ---------------------------------------------------------
plt.figure()
for i in range(len(results)):
    t = np.array(results[i]['t'])
    f_sys = np.array(results[i]['f_sys'])
    f_pll = np.array(results[i]['f_pll'])

    # plot system freq (solid) and pll freq (dashed) using same case color
    plt.plot(t, f_sys, color=colors[i], linestyle='-', linewidth=LW, label=f'{labels[i]}: f_sys')
    plt.plot(t, f_pll, color=colors[i], linestyle='--', linewidth=LW, label=f'{labels[i]}: f_PLL')

plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.title('PLL Frequency Estimate vs System Frequency')
plt.legend(ncol=2)
plt.grid(True, alpha=0.3)


# ---------------------------------------------------------
# (3) PLL angle tracking vs voltage angle  ✅
# ---------------------------------------------------------
plt.figure()
for i in range(len(results)):
    t = np.array(results[i]['t'])

    theta_pll = np.unwrap(np.array(results[i]['theta_pll']))
    theta_v = np.unwrap(np.array(results[i]['theta_v']))

    # relative angles (start at 0)
    theta_pll = theta_pll - theta_pll[0]
    theta_v = theta_v - theta_v[0]

    plt.plot(t, theta_v, color=colors[i], linestyle='-', linewidth=LW, label=f'{labels[i]}: θ_V')
    plt.plot(t, theta_pll, color=colors[i], linestyle='--', linewidth=LW, label=f'{labels[i]}: θ_PLL')

plt.xlabel('Time [s]')
plt.ylabel('Angle [rad]')
plt.title('PLL Angle Tracking vs Voltage Angle (B8)')
plt.legend(ncol=2)
plt.grid(True, alpha=0.3)


# ---------------------------------------------------------
# (4) Angle error (wrapped to [-pi, pi])  ✅
# ---------------------------------------------------------
plt.figure()
for i in range(len(results)):
    t = np.array(results[i]['t'])

    theta_pll = np.unwrap(np.array(results[i]['theta_pll']))
    theta_v = np.unwrap(np.array(results[i]['theta_v']))

    # make them relative first (removes initial offset)
    theta_pll = theta_pll - theta_pll[0]
    theta_v = theta_v - theta_v[0]

    err = wrap_to_pi(theta_pll - theta_v)

    plt.plot(t, err, color=colors[i], linestyle=linestyles[i], linewidth=LW, label=labels[i])

plt.xlabel('Time [s]')
plt.ylabel('Angle error [rad]')
plt.title('PLL Angle Error (wrapped to [-π, π])')
plt.legend()
plt.grid(True, alpha=0.3)


# ---------------------------------------------------------
# (5) dq currents (id, iq)  ✅
# ---------------------------------------------------------
plot_multi('PLL Sensitivity: d-axis current i_d', 'i_d', 'i_d (pu)')
plot_multi('PLL Sensitivity: q-axis current i_q', 'i_q', 'i_q (pu)')

# Zoom near fault for currents
plot_multi('i_d (Zoom near fault)', 'i_d', 'i_d (pu)', xlim=(1.9, 2.6))
plot_multi('i_q (Zoom near fault)', 'i_q', 'i_q (pu)', xlim=(1.9, 2.6))


# ---------------------------------------------------------
# (6) VSC P and Q  ✅
# ---------------------------------------------------------
plot_multi('PLL Sensitivity: VSC Active Power P_vsc', 'P_vsc', 'P_vsc [MW]')
plot_multi('PLL Sensitivity: VSC Reactive Power Q_vsc', 'Q_vsc', 'Q_vsc [MW]')

# Zoom near fault for power
plot_multi('P_vsc (Zoom near fault)', 'P_vsc', 'P_vsc [MW]', xlim=(1.9, 2.6))
plot_multi('Q_vsc (Zoom near fault)', 'Q_vsc', 'Q_vsc [MW]', xlim=(1.9, 2.6))


# ---------------------------------------------------------
# (7) dq voltages Vd, Vq (optional but nice) ✅
# ---------------------------------------------------------
plot_multi('PLL dq Voltage: Vd', 'Vd', 'Vd (pu)')
plot_multi('PLL dq Voltage: Vq', 'Vq', 'Vq (pu)')

plt.show()