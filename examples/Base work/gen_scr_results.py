import sys
import numpy as np
import matplotlib.pyplot as plt
import importlib
import copy
from collections import defaultdict

import tops.dynamic as dps
import tops.solvers as dps_sol

# PATHS
sys.path.insert(0, r"D:\Masters REM+\Master Thesis\paper\TOPS-main\TOPS-main\examples\Base work")
sys.path.append(r"D:\Masters REM+\Master Thesis\paper\TOPS-main\TOPS-main\examples\user_models")

import user_lib
import generator_network as model_data


# ----------------------------------------
# SCR (CORRECT)
# ----------------------------------------
def compute_scr(model, n):

    S_base = model['base_mva']

    Z_L56 = Z_L25 = Z_L69 = None
    Z_T1 = Z_T4 = None
    Z_gen = None

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
        if gen[0] == 'G3':
            Z_gen = 1j * gen[12]

    Z_grid = Z_L56 + Z_L25 + Z_T1
    Z_local = Z_L69 + Z_T4 + Z_gen

    if n == 0:
        Z_th = Z_grid
    else:
        Z_local_eq = Z_local / n
        Z_th = 1 / (1 / Z_grid + 1 / Z_local_eq)

    SCR = 1 / abs(Z_th)

    return SCR


# ----------------------------------------
# APPLY GENERATOR CASE (FILTER)
# ----------------------------------------
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


# ----------------------------------------
# SIMULATION
# ----------------------------------------
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
        v = sol.v   # if error → change to sol.y

        # STEP
        gen_names = list(ps.gen['GEN'].par['name'])
        vset = ps.gen['GEN'].v_setp(x, v)

        if t > 1:
            for g in ['G3','G4','G5','G6']:
                if g in gen_names:
                    vset[gen_names.index(g)] = 1.1

        # measurements
        V_B6 = np.abs(v[iB6])
        Q = ps.gen['GEN'].q_e(x, v) * ps.sys_data['s_n']

        res['t'].append(t)
        res['V'].append(V_B6)

        if 'G3' in gen_names:
            res['Qg'].append(Q[gen_names.index('G3')])
        else:
            res['Qg'].append(0.0)

    return res


# ----------------------------------------
# MAIN
# ----------------------------------------
cases = [0,1,2,4]
results = {}

for n in cases:

    importlib.reload(model_data)
    base_model = model_data.load()

    SCR = compute_scr(base_model, n)
    print(f"n={n} → SCR ≈ {SCR:.2f}")

    model_case = apply_generator_case(base_model, n)
    res = run_simulation(model_case, n)

    results[n] = {'SCR':SCR, **res}


# ----------------------------------------
# PLOT
# ----------------------------------------
plt.figure()
for n in results:
    plt.plot(results[n]['t'], results[n]['V'],
             label=f"SCR={results[n]['SCR']:.1f}")
plt.title("Voltage at B6")
plt.legend(); plt.grid()

plt.figure()
for n in results:
    plt.plot(results[n]['t'], results[n]['Qg'],
             label=f"SCR={results[n]['SCR']:.1f}")
plt.title("Q of G3")
plt.legend(); plt.grid()

plt.show()