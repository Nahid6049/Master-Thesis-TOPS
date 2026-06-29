# -*- coding: utf-8 -*-
"""
RjKvill System
Full 8-Output Multi-Case Study
Includes BOTH Generators
"""

import sys
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import tops.dynamic as dps
import tops.solvers as dps_sol
import importlib


# ==========================================================
# PATH FIX
# ==========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
base_work_dir = os.path.dirname(current_dir)
examples_dir = os.path.dirname(base_work_dir)
tops_root = os.path.dirname(examples_dir)

sys.path.append(base_work_dir)
sys.path.append(tops_root)


# ==========================================================
# CASES
# ==========================================================
CASE_LIST = [1, 2, 3, 4]
all_results = {}

for CASE in CASE_LIST:

    print("\n==============================")
    print("Running CASE:", CASE)
    print("==============================")

    # ------------------------------------------------------
    # LOAD NETWORK FRESH EACH CASE
    # ------------------------------------------------------
    import my_network as model_data
    importlib.reload(model_data)
    model = model_data.load()

    import examples.user_models.user_lib as user_lib

    # ------------------------------------------------------
    # PLL SETTINGS
    # ------------------------------------------------------
    if CASE == 4:
        pll_T = 0.005  # very aggressive PLL
    else:
        pll_T = 0.1

    model['pll'] = {'PLL1': [
        ['name', 'T_filter', 'bus'],
        ['PLL1', pll_T, 'B8'],
    ]}

    # ------------------------------------------------------
    # INITIALIZE SYSTEM
    # ------------------------------------------------------
    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_lib)
    ps.init_dyn_sim()

    t_end = 20
    sol = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0,
        ps.x_0.copy(),
        t_end,
        max_step=5e-3
    )

    idx = {'B1':0,'B2':1,'B5':2,'B6':3,
           'B7':4,'B8':5,'B9':6,'B10':7}

    iB6 = idx['B6']
    iB7 = idx['B7']
    iB8 = idx['B8']

    t = 0
    res = defaultdict(list)

    # ------------------------------------------------------
    # SIMULATION LOOP
    # ------------------------------------------------------
    while t < t_end:

        # ---------------- LOAD ----------------
        if t > 10:
            if CASE == 3:
                ps.y_bus_red_mod[iB6, iB6] = -0.4
            else:
                ps.y_bus_red_mod[iB6, iB6] = -0.1
        else:
            ps.y_bus_red_mod[iB6, iB6] = 0

        # ---------------- FAULT ----------------
        if 2 < t < 2.1:
            if CASE == 2:
                ps.y_bus_red_mod[iB8, iB8] = 1000
            else:
                ps.y_bus_red_mod[iB7, iB7] = 1000
        else:
            ps.y_bus_red_mod[iB7, iB7] = 0
            ps.y_bus_red_mod[iB8, iB8] = 0

        # Solver step
        x = sol.y
        sol.step()
        t = sol.t
        v = sol.v

        for mdl in ps.dyn_mdls:
            mdl.reset_outputs()

        # ================= 8 OUTPUTS =================

        SpeedVector = ps.gen['GEN'].speed(x, v)
        frequency = 50 + 50*np.mean(SpeedVector)

        P_gen = ps.gen['GEN'].P_e(x, v)

        pll_freq = 50 + 50*ps.pll['PLL1'].freq_est(x, v)

        P_vsc = ps.vsc['VSC_SI'].p_e(x, v) * ps.sys_data['s_n']
        Q_vsc = ps.vsc['VSC_SI'].q_e(x, v) * ps.sys_data['s_n']

        I_line = ps.y_bus_red_full[iB6, iB7] * (v[iB6] - v[iB7])
        S_line = v[iB6] * np.conj(I_line) * ps.sys_data['s_n']

        V_B8 = np.abs(v[iB8])

        # Store everything
        res['t'].append(t)

        res['frequency'].append(frequency)
        res['pll_freq'].append(pll_freq)

        res['speed1'].append(SpeedVector[0])
        res['speed2'].append(SpeedVector[1])

        res['P_gen1'].append(P_gen[0])
        res['P_gen2'].append(P_gen[1])

        res['P_vsc'].append(P_vsc)
        res['Q_vsc'].append(Q_vsc)

        res['P_line'].append(np.real(S_line))
        res['voltage'].append(V_B8)

    all_results[CASE] = res
    print("Stored CASE:", CASE, "Points:", len(res['t']))


# ==========================================================
# PLOTTING WITH FIXED STYLES
# ==========================================================

styles = {
    1: ('blue', '-'),
    2: ('red', '--'),
    3: ('green', '-.'),
    4: ('black', ':')
}

def plot_compare(key, title, ylabel):
    plt.figure()
    for CASE in CASE_LIST:
        color, style = styles[CASE]
        plt.plot(all_results[CASE]['t'],
                 all_results[CASE][key],
                 linestyle=style,
                 color=color,
                 linewidth=2,
                 label=f'Case {CASE}')
    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)


# ----------- 8 OUTPUTS -----------
plot_compare('frequency', "System Frequency", "Hz")
plot_compare('pll_freq', "PLL Frequency", "Hz")

plot_compare('speed1', "Generator 1 Speed", "pu")
plot_compare('speed2', "Generator 2 Speed", "pu")

plot_compare('P_gen1', "Generator 1 Electrical Power", "MW")
plot_compare('P_gen2', "Generator 2 Electrical Power", "MW")

plot_compare('P_vsc', "VSC Active Power", "MW")
plot_compare('Q_vsc', "VSC Reactive Power", "MVar")

plot_compare('P_line', "Line Power", "MW")
plot_compare('voltage', "Voltage at B8", "pu")

plt.show()