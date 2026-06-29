import sys
import numpy as np
import matplotlib.pyplot as plt
import importlib
import copy
from collections import defaultdict

import tops.dynamic as dps
import tops.solvers as dps_sol

# PATHS
sys.path.append(r"D:\Masters REM+\Master Thesis\paper\TOPS-main\TOPS-main")
sys.path.append(r"D:\Masters REM+\Master Thesis\paper\TOPS-main\TOPS-main\examples\user_models")

import user_lib
import generator_network as model_data


# ----------------------------------------
# SELECT CASE
# ----------------------------------------
def apply_case(model):
    model = copy.deepcopy(model)

    # choose ONE
    keep = {'G1', 'G3'}                    # SCR ~9.8
    # keep = {'G1', 'G3', 'G4'}            # SCR ~13
    # keep = {'G1', 'G3', 'G4', 'G5', 'G6'}  # SCR ~19

    # generators
    header = model['generators']['GEN'][0]
    model['generators']['GEN'] = [header] + [
        g for g in model['generators']['GEN'][1:] if g[0] in keep
    ]

    # gov
    header = model['gov']['HYGOV'][0]
    model['gov']['HYGOV'] = [header] + [
        g for g in model['gov']['HYGOV'][1:] if g[1] in keep
    ]

    # avr
    header = model['avr']['SEXS'][0]
    model['avr']['SEXS'] = [header] + [
        g for g in model['avr']['SEXS'][1:] if g[1] in keep
    ]

    # pss
    header = model['pss']['STAB1'][0]
    model['pss']['STAB1'] = [header] + [
        g for g in model['pss']['STAB1'][1:] if g[1] in keep
    ]

    return model


# ----------------------------------------
# FAULT SIMULATION
# ----------------------------------------
def run_fault_case(t_fault=2.0, t_clear=2.1, t_end=10.0):
    importlib.reload(model_data)
    model = apply_case(model_data.load())

    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_lib)
    ps.power_flow()
    ps.init_dyn_sim()

    sol = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0,
        ps.x_0.copy(),
        t_end,
        max_step=5e-3
    )

    idx = {'B1': 0, 'B2': 1, 'B5': 2, 'B6': 3, 'B7': 4, 'B8': 5, 'B9': 6, 'B10': 7}
    iB6 = idx['B6']

    gen_names = list(ps.gen['GEN'].par['name'])
    iG1 = gen_names.index('G1')
    iG3 = gen_names.index('G3')

    y_fault = 1e4
    res = defaultdict(list)

    t = 0.0
    while t < t_end:
        # apply fault at B6
        if t_fault < t < t_clear:
            ps.y_bus_red_mod[iB6, iB6] = y_fault
        else:
            ps.y_bus_red_mod[iB6, iB6] = 0.0

        sol.step()

        t = sol.t
        x = sol.y
        v = sol.v

        angles = ps.gen['GEN'].angle(x, v)
        delta = angles[iG3] - angles[iG1]
        speed = ps.gen['GEN'].speed(x, v)[iG3]
        V_B6 = np.abs(v[iB6])

        res['t'].append(t)
        res['angle'].append(delta)
        res['speed'].append(speed)
        res['V_B6'].append(V_B6)

    return res


# ----------------------------------------
# RUN
# ----------------------------------------
res = run_fault_case(t_fault=2.0, t_clear=2.1, t_end=10.0)


# ----------------------------------------
# PLOTS
# ----------------------------------------
plt.figure()
plt.plot(res['t'], res['V_B6'])
plt.axvspan(2.0, 2.1, color='red', alpha=0.2)
plt.title("Voltage at B6")
plt.xlabel("Time [s]")
plt.ylabel("Voltage [pu]")
plt.grid()

plt.figure()
plt.plot(res['t'], res['angle'])
plt.axvspan(2.0, 2.1, color='red', alpha=0.2)
plt.title("Rotor Angle (G3 - G1)")
plt.xlabel("Time [s]")
plt.ylabel("Angle [rad]")
plt.grid()

plt.figure()
plt.plot(res['t'], res['speed'])
plt.axvspan(2.0, 2.1, color='red', alpha=0.2)
plt.title("Rotor Speed (G3)")
plt.xlabel("Time [s]")
plt.ylabel("Speed [pu]")
plt.grid()

plt.show()