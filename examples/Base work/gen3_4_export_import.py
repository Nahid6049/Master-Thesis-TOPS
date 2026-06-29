# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
import importlib
import copy
from collections import defaultdict

import tops.dynamic as dps
import tops.solvers as dps_sol

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_PATH = r"D:\Masters REM+\Master Thesis\paper\TOPS-main\TOPS-main"

sys.path.insert(0, BASE_PATH + r"\examples\Base work")
sys.path.append(BASE_PATH + r"\examples\user_models")

# --------------------------------------------------
# IMPORT
# --------------------------------------------------
import user_lib
import generator_network as model_data


# ----------------------------------------
# CASE 2: G1 + G3 + G4
# ----------------------------------------
def apply_two_gen(model):

    model = copy.deepcopy(model)

    keep = {'G1', 'G3', 'G4'}

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
# SET POWER (G3 ONLY)
# ----------------------------------------
def set_power(model, P_g3):

    model = copy.deepcopy(model)

    header = model['generators']['GEN'][0]
    P_idx = header.index('P')

    for g in model['generators']['GEN'][1:]:
        if g[0] == 'G3':
            g[P_idx] = P_g3

    return model


# ----------------------------------------
# SIMULATION
# ----------------------------------------
def run_simulation(model):

    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_lib)

    ps.power_flow()
    ps.init_dyn_sim()

    sol = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0,
        ps.x_0.copy(),
        10,
        max_step=2e-3
    )

    idx = {'B1':0,'B2':1,'B5':2,'B6':3,'B7':4,'B8':5,'B9':6,'B10':7}
    iB6 = idx['B6']

    res = defaultdict(list)
    t = 0.0

    while t < 10:

        sol.step()

        t = sol.t
        x = sol.y
        v = sol.v

        gen_names = list(ps.gen['GEN'].par['name'])
        vset = ps.gen['GEN'].v_setp(x, v)

        # ----------------------------------------
        # ✅ APPLY STEP ONLY ONCE (CRITICAL FIX)
        # ----------------------------------------
        if t > 1 and not hasattr(ps, "step_done"):

            for g in ['G3', 'G4']:
                if g in gen_names:
                    idx_g = gen_names.index(g)
                    vset[idx_g] = 1.1

            ps.step_done = True

        # ----------------------------------------
        # MEASUREMENTS
        # ----------------------------------------
        V_B6 = np.abs(v[iB6])
        Q = ps.gen['GEN'].q_e(x, v) * ps.sys_data['s_n']
        P = ps.gen['GEN'].p_e(x, v) * ps.sys_data['s_n']

        res['t'].append(t)
        res['V'].append(V_B6)

        if 'G3' in gen_names:
            idx_g3 = gen_names.index('G3')
            res['Pg'].append(P[idx_g3])
            res['Qg'].append(Q[idx_g3])
        else:
            res['Pg'].append(0.0)
            res['Qg'].append(0.0)

        # safety
        if np.any(np.isnan(x)):
            print("⚠️ unstable → stop")
            break

    return res


# ----------------------------------------
# MAIN
# ----------------------------------------
power_cases = {
    'export': 600,
    'balance': 300,
    'import': 100
}

results = {}

for label, Pval in power_cases.items():

    importlib.reload(model_data)
    base_model = model_data.load()

    model_case = apply_two_gen(base_model)
    model_case = set_power(model_case, Pval)

    print(f"\nRunning CASE 2: {label}")

    results[label] = run_simulation(model_case)


# ----------------------------------------
# PLOTS
# ----------------------------------------

# Active Power
plt.figure()
for k in results:
    plt.plot(results[k]['t'], results[k]['Pg'], label=k)
plt.title("Active Power of G3 (Case 2)")
plt.xlabel("Time [s]")
plt.ylabel("MW")
plt.legend()
plt.grid()

# Voltage
plt.figure()
for k in results:
    plt.plot(results[k]['t'], results[k]['V'], label=k)
plt.title("Voltage at B6 (Case 2)")
plt.xlabel("Time [s]")
plt.ylabel("pu")
plt.legend()
plt.grid()

# Reactive Power
plt.figure()
for k in results:
    plt.plot(results[k]['t'], results[k]['Qg'], label=k)
plt.title("Reactive Power of G3 (Case 2)")
plt.xlabel("Time [s]")
plt.ylabel("MVAr")
plt.legend()
plt.grid()

plt.show()