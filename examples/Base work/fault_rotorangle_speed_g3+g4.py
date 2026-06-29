import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import importlib
import copy

import tops.dynamic as dps
import tops.solvers as dps_sol

# PATHS
sys.path.append(r"D:\Masters REM+\Master Thesis\paper\TOPS-main\TOPS-main")
sys.path.append(r"D:\Masters REM+\Master Thesis\paper\TOPS-main\TOPS-main\examples\user_models")

import user_lib
import generator_network as model_data


# ----------------------------------------
# APPLY SCR ≈ 13 (G3 + G4)
# ----------------------------------------
def apply_scr13_case(model):

    model = copy.deepcopy(model)

    keep = {'G1', 'G3', 'G4'}

    # generators
    header = model['generators']['GEN'][0]
    model['generators']['GEN'] = [header] + [g for g in model['generators']['GEN'][1:] if g[0] in keep]

    # gov
    header = model['gov']['HYGOV'][0]
    model['gov']['HYGOV'] = [header] + [g for g in model['gov']['HYGOV'][1:] if g[1] in keep]

    # avr
    header = model['avr']['SEXS'][0]
    model['avr']['SEXS'] = [header] + [g for g in model['avr']['SEXS'][1:] if g[1] in keep]

    # pss
    header = model['pss']['STAB1'][0]
    model['pss']['STAB1'] = [header] + [g for g in model['pss']['STAB1'][1:] if g[1] in keep]

    return model


# ----------------------------------------
# FAULT SIMULATION
# ----------------------------------------
def run_fault_case(t_fault=2.0, t_clear=2.1):

    importlib.reload(model_data)
    base_model = model_data.load()

    model = apply_scr13_case(base_model)

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

    gen_names = list(ps.gen['GEN'].par['name'])
    idx_g3 = gen_names.index('G3')
    idx_g1 = gen_names.index('G1')

    y_fault = 1e4

    res = defaultdict(list)

    t = 0

    while t < 10:

        # APPLY FAULT
        if t_fault < t < t_clear:
            ps.y_bus_red_mod[iB6, iB6] = y_fault
        else:
            ps.y_bus_red_mod[iB6, iB6] = 0

        sol.step()

        t = sol.t
        x = sol.y
        v = sol.v

        # -------------------------
        # MEASUREMENTS
        # -------------------------
        angles = ps.gen['GEN'].angle(x, v)
        delta = angles[idx_g3] - angles[idx_g1]

        speed = ps.gen['GEN'].speed(x, v)[idx_g3]

        V_B6 = np.abs(v[iB6])

        res['t'].append(t)
        res['angle'].append(delta)
        res['speed'].append(speed)
        res['V_B6'].append(V_B6)

    return res


# ----------------------------------------
# RUN
# ----------------------------------------
res = run_fault_case(t_fault=2.0, t_clear=2.1)


# ----------------------------------------
# PLOTS
# ----------------------------------------

# Voltage
plt.figure()
plt.plot(res['t'], res['V_B6'])
plt.axvspan(2,2.1,color='red',alpha=0.2)
plt.title("Voltage at B6 (SCR ≈ 13)")
plt.xlabel("Time [s]")
plt.ylabel("Voltage [pu]")
plt.grid()

# Rotor Angle
plt.figure()
plt.plot(res['t'], res['angle'])
plt.axvspan(2,2.1,color='red',alpha=0.2)
plt.title("Rotor Angle (G3 - G1)")
plt.xlabel("Time [s]")
plt.ylabel("Angle [rad]")
plt.grid()

# Rotor Speed
plt.figure()
plt.plot(res['t'], res['speed'])
plt.axvspan(2,2.1,color='red',alpha=0.2)
plt.title("G3 Speed")
plt.xlabel("Time [s]")
plt.ylabel("Speed [pu]")
plt.grid()

plt.show()