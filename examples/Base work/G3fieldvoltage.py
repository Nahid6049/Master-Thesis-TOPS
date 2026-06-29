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
# PATHS (UPDATE IF NEEDED)
# --------------------------------------------------
BASE_PATH = r"D:\Masters REM+\Master Thesis\paper\TOPS-main\TOPS-main"

sys.path.insert(0, BASE_PATH + r"\examples\Base work")
sys.path.append(BASE_PATH + r"\examples\user_models")

# --------------------------------------------------
# IMPORT MODEL
# --------------------------------------------------
import user_lib
import generator_network as model_data


# --------------------------------------------------
# SCR CALCULATION (WITH G1 INCLUDED)
# --------------------------------------------------
def compute_scr(model, n):

    Z_L56 = Z_L25 = Z_L69 = None
    Z_T1 = Z_T4 = None
    Z_gen = None
    Z_G1 = None

    # --------------------
    # LINES
    # --------------------
    for line in model['lines'][1:]:
        Z = line[7] + 1j * line[8]

        if line[0] == 'L5-6':
            Z_L56 = Z
        elif line[0] == 'L2-5':
            Z_L25 = Z
        elif line[0] == 'L6-9':
            Z_L69 = Z

    # --------------------
    # TRANSFORMERS
    # --------------------
    for tr in model['transformers'][1:]:
        Z = tr[6] + 1j * tr[7]

        if tr[0] == 'T1':
            Z_T1 = Z
        elif tr[0] == 'T4':
            Z_T4 = Z

    # --------------------
    # GENERATORS
    # --------------------
    for gen in model['generators']['GEN'][1:]:

        name = gen[0]

        # G3 (local generator)
        if name == 'G3':
            Z_gen = 1j * gen[12]   # Xd''

        # G1 (slack generator INCLUDED)
        if name == 'G1':
            Z_G1 = 1j * gen[12]   # Xd''

    # --------------------
    # IMPEDANCES
    # --------------------
    Z_grid = Z_L56 + Z_L25 + Z_T1 + Z_G1
    Z_local = Z_L69 + Z_T4 + Z_gen

    if n == 0:
        Z_th = Z_grid
    else:
        Z_local_eq = Z_local / n
        Z_th = 1 / (1 / Z_grid + 1 / Z_local_eq)

    SCR = 1 / abs(Z_th)

    return SCR


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

    # generators
    header = model['generators']['GEN'][0]
    rows = [g for g in model['generators']['GEN'][1:] if g[0] in keep]
    model['generators']['GEN'] = [header] + rows

    # gov
    header = model['gov']['HYGOV'][0]
    rows = [g for g in model['gov']['HYGOV'][1:] if g[1] in keep]
    model['gov']['HYGOV'] = [header] + rows

    # avr
    header = model['avr']['SEXS'][0]
    rows = [g for g in model['avr']['SEXS'][1:] if g[1] in keep]
    model['avr']['SEXS'] = [header] + rows

    # pss
    header = model['pss']['STAB1'][0]
    rows = [g for g in model['pss']['STAB1'][1:] if g[1] in keep]
    model['pss']['STAB1'] = [header] + rows

    return model


# --------------------------------------------------
# SIMULATION
# --------------------------------------------------
def run_simulation(model, n):

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

        # Voltage step
        vset = ps.gen['GEN'].v_setp(x, v)

        if t > 1:
            for g in ['G3','G4','G5','G6']:
                if g in gen_names:
                    vset[gen_names.index(g)] = 1.1

        # Measurements
        V_B6 = np.abs(v[iB6])
        Q = ps.gen['GEN'].q_e(x, v) * ps.sys_data['s_n']

        try:
            Efd_all = ps.avr['SEXS'].output(x, v)
        except:
            Efd_all = ps.avr['SEXS'].u(x, v)

        res['t'].append(t)
        res['V'].append(V_B6)

        if 'G3' in gen_names:
            idx_g3 = gen_names.index('G3')
            res['Qg'].append(Q[idx_g3])
            res['Efd'].append(Efd_all[idx_g3])
        else:
            res['Qg'].append(0.0)
            res['Efd'].append(0.0)

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
    print(f"n={n} → SCR ≈ {SCR:.2f}")

    model_case = apply_generator_case(base_model, n)
    res = run_simulation(model_case, n)

    results[n] = {'SCR': SCR, **res}


# --------------------------------------------------
# PLOTS
# --------------------------------------------------

plt.figure()
for n in results:
    plt.plot(results[n]['t'], results[n]['V'],
             label=f"SCR={results[n]['SCR']:.1f}")
plt.title("Voltage at B6")
plt.legend()
plt.grid()

plt.figure()
for n in results:
    plt.plot(results[n]['t'], results[n]['Qg'],
             label=f"SCR={results[n]['SCR']:.1f}")
plt.title("Reactive Power of G3")
plt.legend()
plt.grid()

plt.figure()
for n in results:
    plt.plot(results[n]['t'], results[n]['Efd'],
             label=f"SCR={results[n]['SCR']:.1f}")
plt.title("Field Voltage of G3")
plt.legend()
plt.grid()

plt.show()