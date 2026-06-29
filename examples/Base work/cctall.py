# ===== SCR ≈ 13 (FINAL CLEAN VERSION) =====

import sys
import numpy as np
import matplotlib.pyplot as plt
import importlib
import copy
from collections import defaultdict

import tops.dynamic as dps
import tops.solvers as dps_sol

# ----------------------------------------
# PATHS
# ----------------------------------------
BASE_PATH = r"D:\Masters REM+\Master Thesis\paper\TOPS-main\TOPS-main"

sys.path.insert(0, BASE_PATH + r"\examples\Base work")
sys.path.append(BASE_PATH + r"\examples\user_models")

import user_lib
import generator_network as model_data


# ----------------------------------------
# APPLY CASE (SCR ≈ 13)
# ----------------------------------------
def apply_case(model):

    model = copy.deepcopy(model)

    keep = {'G1','G3','G4'}

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

    # pss → only G1
    header = model['pss']['STAB1'][0]
    model['pss']['STAB1'] = [header] + [
        g for g in model['pss']['STAB1'][1:] if g[1] == 'G1'
    ]

    return model


# ----------------------------------------
# SIMULATION
# ----------------------------------------
def simulate(model, t_clear):

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
    iG3 = gen_names.index('G3')
    iG1 = gen_names.index('G1')

    res = defaultdict(list)

    t = 0
    y_fault = 1e4
    stable = True

    while t < 10:

        # fault
        if 2 < t < t_clear:
            ps.y_bus_red_mod[iB6, iB6] = y_fault
        else:
            ps.y_bus_red_mod[iB6, iB6] = 0

        sol.step()

        t = sol.t
        x = sol.y
        v = sol.v

        # rotor angle FIRST (important)
        angles = ps.gen['GEN'].angle(x, v)
        delta = angles[iG3] - angles[iG1]

        if abs(delta) > np.pi:
            stable = False

        # then numerical check
        if np.any(np.isnan(x)):
            stable = False
            break

        # reactive power
        Q = ps.gen['GEN'].q_e(x, v) * ps.sys_data['s_n']
        Qg3 = Q[iG3]

        res['t'].append(t)
        res['angle'].append(delta)
        res['Qg'].append(Qg3)

    return stable, res


# ----------------------------------------
# FIND CCT
# ----------------------------------------
def find_cct(model):

    last = None

    for t_clear in np.arange(2.0, 4.0, 0.05):

        stable, _ = simulate(model, t_clear)

        print(f"{t_clear:.2f} → {'STABLE' if stable else 'UNSTABLE'}")

        if stable:
            last = t_clear
        else:
            break

    return last


# ----------------------------------------
# MAIN
# ----------------------------------------
importlib.reload(model_data)
base_model = model_data.load()

model = apply_case(base_model)

# find CCT
cct = find_cct(model)

print(f"\n👉 CCT ≈ {cct:.2f} s")

# simulate near CCT
stable_res = simulate(model, cct)[1]
unstable_res = simulate(model, cct + 0.05)[1]


# ----------------------------------------
# PLOTS
# ----------------------------------------

# Rotor angle
plt.figure()
plt.plot(stable_res['t'], stable_res['angle'], label=f"Stable ({cct:.2f}s)")
plt.plot(unstable_res['t'], unstable_res['angle'], '--', label=f"Unstable ({cct+0.05:.2f}s)")
plt.axvspan(2, cct, color='green', alpha=0.1)
plt.axvspan(2, cct+0.05, color='red', alpha=0.1)
plt.title("Rotor Angle (SCR ≈ 13)")
plt.xlabel("Time [s]")
plt.ylabel("Angle [rad]")
plt.legend()
plt.grid()


# Reactive power
plt.figure()
plt.plot(stable_res['t'], stable_res['Qg'], label=f"Stable ({cct:.2f}s)")
plt.plot(unstable_res['t'], unstable_res['Qg'], '--', label=f"Unstable ({cct+0.05:.2f}s)")
plt.axvspan(2, cct, color='green', alpha=0.1)
plt.axvspan(2, cct+0.05, color='red', alpha=0.1)
plt.title("Reactive Power of G3 (SCR ≈ 13)")
plt.xlabel("Time [s]")
plt.ylabel("MVAr")
plt.legend()
plt.grid()

plt.show()