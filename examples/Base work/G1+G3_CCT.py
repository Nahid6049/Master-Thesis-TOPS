# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
import importlib
import copy

import tops.dynamic as dps
import tops.solvers as dps_sol

# --------------------------------------------------
# PATH
# --------------------------------------------------
BASE_PATH = r"D:\Masters REM+\Master Thesis\paper\TOPS-main\TOPS-main"

sys.path.insert(0, BASE_PATH + r"\examples\Base work")
sys.path.append(BASE_PATH + r"\examples\user_models")

import user_lib
import generator_network as model_data


# --------------------------------------------------
# APPLY G1 + G3
# --------------------------------------------------
def apply_case(model):
    model = copy.deepcopy(model)

    keep = {'G1', 'G3'}

    header = model['generators']['GEN'][0]
    model['generators']['GEN'] = [header] + [
        g for g in model['generators']['GEN'][1:] if g[0] in keep
    ]

    header = model['gov']['HYGOV'][0]
    model['gov']['HYGOV'] = [header] + [
        g for g in model['gov']['HYGOV'][1:] if g[1] in keep
    ]

    header = model['avr']['SEXS'][0]
    model['avr']['SEXS'] = [header] + [
        g for g in model['avr']['SEXS'][1:] if g[1] in keep
    ]

    header = model['pss']['STAB1'][0]
    model['pss']['STAB1'] = [header] + [
        g for g in model['pss']['STAB1'][1:] if g[1] in keep
    ]

    return model


# --------------------------------------------------
# SET SAME PU LOADING (IMPORTANT)
# --------------------------------------------------
def set_power(model):
    model = copy.deepcopy(model)

    header = model['generators']['GEN'][0]
    P_idx = header.index('P')

    for row in model['generators']['GEN'][1:]:
        if row[0] == 'G3':
            row[P_idx] = 300  # ~0.85 pu

    return model


# --------------------------------------------------
# SAFE VSC READ
# --------------------------------------------------
def read_vsc(ps, x, v):

    if 'VSC_SI' not in ps.vsc:
        return 0.0, 0.0

    vsc = ps.vsc['VSC_SI']

    try:
        P = float(np.asarray(vsc.p_e(x, v)).item()) * ps.sys_data['s_n']
    except:
        P = 0.0

    try:
        Q = float(np.asarray(vsc.q_e(x, v)).item()) * ps.sys_data['s_n']
    except:
        Q = 0.0

    return P, Q


# --------------------------------------------------
# CCT SIMULATION
# --------------------------------------------------
def simulate_cct(t_clear):

    importlib.reload(model_data)
    model = model_data.load()

    model = apply_case(model)
    model = set_power(model)

    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_lib)

    ps.power_flow()
    ps.init_dyn_sim()

    sol = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0,
        ps.x_0.copy(),
        5,
        max_step=2e-3
    )

    res = {'t': [], 'angle': [], 'speed': [], 'Qvsc': []}

    t = 0

    while t < 5:

        sol.step()

        t = sol.t
        x = sol.y
        v = sol.v

        gen_names = list(ps.gen['GEN'].par['name'])

        # ---------------- FAULT ----------------
        if 1.0 <= t <= t_clear:
            ps.y_bus_red_mod[(3, 3)] = 1e6  # B6 fault
        else:
            ps.y_bus_red_mod[(3, 3)] = 0

        # ---------------- ANGLE ----------------
        angles = ps.gen['GEN'].angle(x, v)

        if 'G1' in gen_names and 'G3' in gen_names:
            delta = angles[gen_names.index('G3')] - angles[gen_names.index('G1')]
        else:
            delta = 0

        # ---------------- SPEED ----------------
        speed = ps.gen['GEN'].speed(x, v)
        w = speed[gen_names.index('G3')] if 'G3' in gen_names else 0

        # ---------------- VSC ----------------
        _, Qvsc = read_vsc(ps, x, v)

        # store
        res['t'].append(t)
        res['angle'].append(delta)
        res['speed'].append(w)
        res['Qvsc'].append(Qvsc)

        # ---------------- INSTABILITY CHECK ----------------

        # 1. large angle
        if abs(delta) > 3.0:
            return res, False

        # 2. divergence trend
        if len(res['angle']) > 50:
            if abs(res['angle'][-1]) > abs(res['angle'][-20]) + 0.2:
                return res, False

    return res, True


# --------------------------------------------------
# FIND CCT
# --------------------------------------------------
print("\nG1 + G3 CCT TEST\n")

for t_clear in np.arange(1.05, 1.12, 0.01):

    res, stable = simulate_cct(t_clear)

    print(f"t_clear = {t_clear:.3f} s -> {'STABLE' if stable else 'UNSTABLE'}")

    if not stable:
        break


# --------------------------------------------------
# PLOTS
# --------------------------------------------------
res_stable, _ = simulate_cct(1.07)
res_unstable, _ = simulate_cct(1.09)

# Rotor angle
plt.figure()
plt.plot(res_stable['t'], res_stable['angle'])
plt.title("Rotor Angle (Stable)")
plt.grid()

plt.figure()
plt.plot(res_unstable['t'], res_unstable['angle'])
plt.title("Rotor Angle (Unstable)")
plt.grid()

# Speed
plt.figure()
plt.plot(res_stable['t'], res_stable['speed'])
plt.title("Generator Speed (Stable)")
plt.grid()

# VSC Q
plt.figure()
plt.plot(res_stable['t'], res_stable['Qvsc'])
plt.title("VSC Reactive Power")
plt.grid()

plt.show()