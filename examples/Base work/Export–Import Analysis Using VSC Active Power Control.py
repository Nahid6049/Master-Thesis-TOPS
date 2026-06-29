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

import user_lib
import generator_network as model_data


# --------------------------------------------------
# SCR
# --------------------------------------------------
def compute_scr(model, n):

    Z_L56 = Z_L25 = Z_L69 = None
    Z_T1 = Z_T4 = None
    Z_G1 = None
    Z_G3 = None

    for line in model['lines'][1:]:
        Z = line[7] + 1j * line[8]
        if line[0] == 'L5-6': Z_L56 = Z
        elif line[0] == 'L2-5': Z_L25 = Z
        elif line[0] == 'L6-9': Z_L69 = Z

    for tr in model['transformers'][1:]:
        Z = tr[6] + 1j * tr[7]
        if tr[0] == 'T1': Z_T1 = Z
        elif tr[0] == 'T4': Z_T4 = Z

    for gen in model['generators']['GEN'][1:]:
        if gen[0] == 'G1':
            Z_G1 = 1j * gen[12]
        if gen[0] == 'G3':
            Z_G3 = 1j * gen[12]

    Z_grid = Z_L25 + Z_L56 + Z_T1 + Z_G1
    Z_local = Z_L69 + Z_T4 + Z_G3

    if n == 0:
        Z_th = Z_grid
    else:
        Z_local_eq = Z_local / n
        Z_th = 1 / (1 / Z_grid + 1 / Z_local_eq)

    return 1 / abs(Z_th)


# --------------------------------------------------
# APPLY GENERATOR CASE
# --------------------------------------------------
def apply_generator_case(model, n):

    model = copy.deepcopy(model)

    if n == 0:
        keep = {'G1'}
    elif n == 1:
        keep = {'G1', 'G3'}
    elif n == 2:
        keep = {'G1', 'G3', 'G4'}
    elif n == 4:
        keep = {'G1', 'G3', 'G4', 'G5', 'G6'}

    model['generators']['GEN'] = [model['generators']['GEN'][0]] + [
        g for g in model['generators']['GEN'][1:] if g[0] in keep
    ]

    model['gov']['HYGOV'] = [model['gov']['HYGOV'][0]] + [
        g for g in model['gov']['HYGOV'][1:] if g[1] in keep
    ]

    model['avr']['SEXS'] = [model['avr']['SEXS'][0]] + [
        g for g in model['avr']['SEXS'][1:] if g[1] in keep
    ]

    model['pss']['STAB1'] = [model['pss']['STAB1'][0]] + [
        g for g in model['pss']['STAB1'][1:] if g[1] in keep
    ]

    return model


# --------------------------------------------------
# COMPUTE LOAD
# --------------------------------------------------
def compute_total_load(model):
    P_total = 0.0
    for load in model['loads'][1:]:
        P_total += load[5]
    return P_total


# --------------------------------------------------
# DISTRIBUTE GENERATION
# --------------------------------------------------
def set_generation_equal_to_load(model):

    model = copy.deepcopy(model)

    P_load = compute_total_load(model)

    local_gens = [g for g in model['generators']['GEN'][1:] if g[0] != 'G1']
    n_local = len(local_gens)

    if n_local == 0:
        return model

    P_per_gen = P_load / n_local

    for gen in local_gens:
        gen[5] = P_per_gen

    return model


# --------------------------------------------------
# SIMULATION
# --------------------------------------------------
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
        max_step=5e-3
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

        # voltage step
        vset = ps.gen['GEN'].v_setp(x, v)
        if t > 1:
            for g in gen_names:
                vset[gen_names.index(g)] = 1.1

        V_B6 = np.abs(v[iB6])
        Q = ps.gen['GEN'].q_e(x, v) * ps.sys_data['s_n']

        try:
            Efd = ps.avr['SEXS'].output(x, v)
        except:
            Efd = ps.avr['SEXS'].u(x, v)

        res['t'].append(t)
        res['V'].append(V_B6)
        res['Qg_all'].append(Q.copy())
        res['Efd_all'].append(Efd.copy())
        res['gen_names'] = gen_names

    return res


# --------------------------------------------------
# MAIN
# --------------------------------------------------
cases = [0, 1, 2, 4]
results = {}

for n in cases:

    importlib.reload(model_data)
    base_model = model_data.load()

    SCR = compute_scr(base_model, n)

    model_case = apply_generator_case(base_model, n)

    # ✔ BALANCED GENERATION = LOAD
    model_case = set_generation_equal_to_load(model_case)

    res = run_simulation(model_case)

    results[n] = {'SCR': SCR, **res}

    print(f"n={n} → SCR={SCR:.2f}")


# --------------------------------------------------
# PLOT
# --------------------------------------------------
plt.figure()
for n in results:
    plt.plot(results[n]['t'], results[n]['V'],
             label=f"SCR={results[n]['SCR']:.1f}")

plt.title("Voltage at B6 (Stable Operation)")
plt.legend()
plt.grid()

plt.show()