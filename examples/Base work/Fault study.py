import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import importlib
import tops.dynamic as dps
import tops.solvers as dps_sol

# PATHS
sys.path.append(r"D:\Masters REM+\Master Thesis\paper\TOPS-main\TOPS-main")
sys.path.append(r"D:\Masters REM+\Master Thesis\paper\TOPS-main\TOPS-main\examples\user_models")

import user_lib
import my_network as model_data


def run_fault_case():

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

    idx = {'B1':0,'B2':1,'B5':2,'B6':3,'B7':4,'B8':5,'B9':6,'B10':7}

    iB6 = idx['B6']
    iB8 = idx['B8']

    # locate generator rotor angle states
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

        # fault at B8
        if 2 < t < 2.1:
            ps.y_bus_red_mod[iB8, iB8] = y_fault
        else:
            ps.y_bus_red_mod[iB8, iB8] = 0

        sol.step()

        t = sol.t
        x = sol.y
        v = sol.v

        V_B6 = np.abs(v[iB6])
        V_B8 = np.abs(v[iB8])

        Q_vsc = ps.vsc['VSC_SI'].q_e(x, v) * ps.sys_data['s_n']
        Q_gen = ps.gen['GEN'].q_e(x, v) * ps.sys_data['s_n']

        # relative rotor angle
        delta = x[angle_states[1]] - x[angle_states[0]]

        # generator speed
        speed = ps.gen['GEN'].speed(x, v)

        res['t'].append(t)
        res['V_B6'].append(V_B6)
        res['V_B8'].append(V_B8)
        res['Q_vsc'].append(Q_vsc)
        res['Q_gen'].append(Q_gen[1])
        res['angle'].append(delta)
        res['speed'].append(speed[1])

    return res


res = run_fault_case()


# -------- PLOTS --------

plt.figure()
plt.plot(res['t'], res['V_B6'])
plt.axvspan(2,2.1,color='red',alpha=0.2)
plt.title("Voltage B6")
plt.xlabel("Time [s]")
plt.ylabel("Voltage [pu]")
plt.grid()

plt.figure()
plt.plot(res['t'], res['V_B8'])
plt.axvspan(2,2.1,color='red',alpha=0.2)
plt.title("Voltage B8 (Fault Bus)")
plt.xlabel("Time [s]")
plt.ylabel("Voltage [pu]")
plt.grid()

plt.figure()
plt.plot(res['t'], res['Q_gen'])
plt.axvspan(2,2.1,color='red',alpha=0.2)
plt.title("Generator Reactive Power")
plt.xlabel("Time [s]")
plt.ylabel("Q [MVar]")
plt.grid()

plt.figure()
plt.plot(res['t'], res['Q_vsc'])
plt.axvspan(2,2.1,color='red',alpha=0.2)
plt.title("VSC Reactive Power")
plt.xlabel("Time [s]")
plt.ylabel("Q [MVar]")
plt.grid()

plt.figure()
plt.plot(res['t'], res['angle'])
plt.axvspan(2,2.1,color='red',alpha=0.2)
plt.title("Relative Rotor Angle (G3 − G1)")
plt.xlabel("Time [s]")
plt.ylabel("Angle [rad]")
plt.grid()

plt.figure()
plt.plot(res['t'], res['speed'])
plt.axvspan(2,2.1,color='red',alpha=0.2)
plt.title("Generator Speed / Frequency")
plt.xlabel("Time [s]")
plt.ylabel("Speed [pu]")
plt.grid()

plt.show()