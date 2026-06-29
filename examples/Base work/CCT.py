# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import importlib
import tops.dynamic as dps
import tops.solvers as dps_sol

# -------------------------------------------------
# PATHS
# -------------------------------------------------

sys.path.append(r"D:\Masters REM+\Master Thesis\paper\TOPS-main\TOPS-main")
sys.path.append(r"D:\Masters REM+\Master Thesis\paper\TOPS-main\TOPS-main\examples\user_models")

import user_lib
import my_network as model_data


# -------------------------------------------------
# SIMULATION FUNCTION
# -------------------------------------------------

def run_fault_case(clear_time):

    importlib.reload(model_data)
    model = model_data.load()

    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_lib)

    ps.power_flow()
    ps.init_dyn_sim()

    sol = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0,
        ps.x_0.copy(),
        10,
        max_step=5e-3
    )

    # Bus index
    idx = {'B1':0,'B2':1,'B5':2,'B6':3,'B7':4,'B8':5,'B9':6,'B10':7}

    iB6 = idx['B6']
    iB8 = idx['B8']

    # find generator rotor angles
    angle_states = []

    for i, desc in enumerate(ps.state_desc):
        if desc[1] == 'angle':
            angle_states.append(i)

    y_fault = 10000

    res = defaultdict(list)

    t = 0

    while t < 10:

        x = sol.y
        v = sol.v

        # generator voltage step
       

        # fault at B8
        if 2 < t < clear_time:
            ps.y_bus_red_mod[iB8, iB8] = y_fault
        else:
            ps.y_bus_red_mod[iB8, iB8] = 0

        sol.step()

        t = sol.t
        x = sol.y
        v = sol.v

        # voltages
        V_B6 = np.abs(v[iB6])
        V_B8 = np.abs(v[iB8])

        # reactive power
        Q_vsc = ps.vsc['VSC_SI'].q_e(x, v) * ps.sys_data['s_n']
        Q_gen = ps.gen['GEN'].q_e(x, v) * ps.sys_data['s_n']

        # rotor angle difference
        delta = x[angle_states[1]] - x[angle_states[0]]

        res['t'].append(t)
        res['V_B6'].append(V_B6)
        res['V_B8'].append(V_B8)
        res['Q_vsc'].append(Q_vsc)
        res['Q_gen'].append(Q_gen[1])
        res['angle'].append(delta)

    return res


# -------------------------------------------------
# FAULT CLEARING TIMES
# -------------------------------------------------

case1 = run_fault_case(2.3)
case2 = run_fault_case(2.4)
case3 = run_fault_case(2.5)
case4 = run_fault_case(2.6)


# -------------------------------------------------
# PLOTS
# -------------------------------------------------

# Voltage B6
plt.figure()
plt.plot(case1['t'], case1['V_B6'], label="Clear 2.05s")
plt.plot(case2['t'], case2['V_B6'], label="Clear 2.10s")
plt.plot(case3['t'], case3['V_B6'], label="Clear 2.15s")
plt.plot(case4['t'], case4['V_B6'], label="Clear 2.20s")
plt.title("Voltage at Bus B6")
plt.xlabel("Time [s]")
plt.ylabel("Voltage [pu]")
plt.legend()
plt.grid()


# Voltage B8
plt.figure()
plt.plot(case1['t'], case1['V_B8'], label="Clear 2.05s")
plt.plot(case2['t'], case2['V_B8'], label="Clear 2.10s")
plt.plot(case3['t'], case3['V_B8'], label="Clear 2.15s")
plt.plot(case4['t'], case4['V_B8'], label="Clear 2.20s")
plt.title("Voltage at Bus B8 (Fault Bus)")
plt.xlabel("Time [s]")
plt.ylabel("Voltage [pu]")
plt.legend()
plt.grid()


# Generator reactive power
plt.figure()
plt.plot(case1['t'], case1['Q_gen'], label="Clear 2.05s")
plt.plot(case2['t'], case2['Q_gen'], label="Clear 2.10s")
plt.plot(case3['t'], case3['Q_gen'], label="Clear 2.15s")
plt.plot(case4['t'], case4['Q_gen'], label="Clear 2.20s")
plt.title("Generator Reactive Power")
plt.xlabel("Time [s]")
plt.ylabel("Q [MVar]")
plt.legend()
plt.grid()


# VSC reactive power
plt.figure()
plt.plot(case1['t'], case1['Q_vsc'], label="Clear 2.05s")
plt.plot(case2['t'], case2['Q_vsc'], label="Clear 2.10s")
plt.plot(case3['t'], case3['Q_vsc'], label="Clear 2.15s")
plt.plot(case4['t'], case4['Q_vsc'], label="Clear 2.20s")
plt.title("VSC Reactive Power")
plt.xlabel("Time [s]")
plt.ylabel("Q [MVar]")
plt.legend()
plt.grid()


# Rotor angle stability
plt.figure()
plt.plot(case1['t'], case1['angle'], label="Clear 2.05s")
plt.plot(case2['t'], case2['angle'], label="Clear 2.10s")
plt.plot(case3['t'], case3['angle'], label="Clear 2.15s")
plt.plot(case4['t'], case4['angle'], label="Clear 2.20s")
plt.title("Rotor Angle Stability (CCT Study)")
plt.xlabel("Time [s]")
plt.ylabel("Angle [rad]")
plt.legend()
plt.grid()

plt.show()