# -*- coding: utf-8 -*-
"""
Synthetic Inertia Sensitivity Study

Cases:
K_SI = 0
K_SI = 0.5
K_SI = 1.0

Disturbances
- Fault at B8 (2–2.1 s)
- Load change at B6 (after 10 s)
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
# PATH FIX
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
base_work_dir = os.path.dirname(current_dir)
examples_dir = os.path.dirname(base_work_dir)
tops_root = os.path.dirname(examples_dir)

sys.path.append(base_work_dir)
sys.path.append(tops_root)

import examples.user_models.user_lib as user_lib


# ---------------------------------------------------------
# CASES
# ---------------------------------------------------------
K_cases = [0, 0.5, 1.0]
labels = ['K_SI = 0', 'K_SI = 0.3', 'K_SI = 0.5']
colors = ['green', 'orange', 'red']


# ---------------------------------------------------------
# SIMULATION FUNCTION
# ---------------------------------------------------------
def run_simulation(K_SI_value):

    import my_network as model_data
    importlib.reload(model_data)
    model = model_data.load()

    # modify inertia gain
    model['vsc']['VSC_SI'][1][13] = K_SI_value

    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_lib)
    ps.init_dyn_sim()

    sol = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0,
        ps.x_0.copy(),
        20,
        max_step=5e-3
    )

    idx = {'B1':0,'B2':1,'B5':2,'B6':3,'B7':4,'B8':5,'B9':6,'B10':7}

    iB6 = idx['B6']
    iB8 = idx['B8']

    res = defaultdict(list)

    t = 0

    while t < 20:

        # Load change
        if t > 10:
            ps.y_bus_red_mod[iB6, iB6] = -0.1
        else:
            ps.y_bus_red_mod[iB6, iB6] = 0

        # Fault at B8
        if 2 < t < 2.1:
            ps.y_bus_red_mod[iB8, iB8] = 1000
        else:
            ps.y_bus_red_mod[iB8, iB8] = 0

        x = sol.y
        sol.step()
        t = sol.t
        v = sol.v

        for mdl in ps.dyn_mdls:
            mdl.reset_outputs()

        # System frequency
        SpeedVector = ps.gen['GEN'].speed(x, v)
        freq = 50 + 50*np.mean(SpeedVector)

        # VSC power
        P_vsc = ps.vsc['VSC_SI'].p_e(x, v) * ps.sys_data['s_n']

        # ROCOF
        rocof = ps.vsc['VSC_SI'].rocof_est(x, v)

        res['t'].append(t)
        res['freq'].append(freq)
        res['P_vsc'].append(P_vsc)
        res['rocof'].append(rocof)

    return res


# ---------------------------------------------------------
# RUN ALL CASES
# ---------------------------------------------------------
results = []

for K in K_cases:
    print("Running simulation K_SI =", K)
    results.append(run_simulation(K))


# ---------------------------------------------------------
# PLOTS
# ---------------------------------------------------------

# Frequency
plt.figure()

for i in range(3):
    t = np.array(results[i]['t'])
    f = np.array(results[i]['freq'])

    plt.plot(t, f, color=colors[i], label=labels[i], linewidth=1)

plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.title('System Frequency Response')
plt.legend()
plt.grid()


# ROCOF
plt.figure()

for i in range(3):
    t = np.array(results[i]['t'])
    r = np.array(results[i]['rocof'])

    plt.plot(t, r, color=colors[i], label=labels[i], linewidth=1)

plt.xlabel('Time [s]')
plt.ylabel('ROCOF [Hz/s]')
plt.title('ROCOF Response')
plt.legend()
plt.grid()


# VSC Active Power
plt.figure()

for i in range(3):
    t = np.array(results[i]['t'])
    P = np.array(results[i]['P_vsc'])

    plt.plot(t, P, color=colors[i], label=labels[i], linewidth=1)

plt.xlabel('Time [s]')
plt.ylabel('VSC Active Power [MW]')
plt.title('Synthetic Inertia Power Injection')
plt.legend()
plt.grid()


plt.show()