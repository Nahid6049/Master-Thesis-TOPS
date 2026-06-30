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
# ONLY STRONG GRID
# --------------------------------------------------
ACTIVE_GENS = ['G1']


# --------------------------------------------------
# EXPORT / IMPORT CASES
# --------------------------------------------------
POWER_CASES = {
    'export': 300,
    'balance': 200,
    'import': 100
}


# --------------------------------------------------
# APPLY GENERATORS
# --------------------------------------------------
def apply_gen_case(model, active_gens):
    model = copy.deepcopy(model)

    header = model['generators']['GEN'][0]
    model['generators']['GEN'] = [header] + [
        g for g in model['generators']['GEN'][1:] if g[0] in active_gens
    ]

    for key in ['gov', 'avr', 'pss']:
        subkey = list(model[key].keys())[0]
        header = model[key][subkey][0]
        model[key][subkey] = [header] + [
            g for g in model[key][subkey][1:] if g[1] in active_gens
        ]

    return model


# --------------------------------------------------
# DISTRIBUTE POWER
# --------------------------------------------------
def distribute_power(model, total_power, active_gens):
    model = copy.deepcopy(model)

    local_gens = [g for g in active_gens if g != 'G1']
    n = len(local_gens)

    if n == 0:
        return model

    share = total_power / n

    header = model['generators']['GEN'][0]
    P_idx = header.index('P')

    for g in model['generators']['GEN'][1:]:
        if g[0] in local_gens:
            g[P_idx] = share

    return model


# --------------------------------------------------
# SAFE VSC READ
# --------------------------------------------------
def read_vsc(ps, x, v):
    if 'VSC_SI' not in ps.vsc:
        return 0.0, 0.0

    vsc = ps.vsc['VSC_SI']
    s_base = ps.sys_data['s_n']

    P = vsc.p_e(x, v) * s_base
    Q = vsc.q_e(x, v) * s_base

    return float(np.asarray(P).item()), float(np.asarray(Q).item())


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

        # Voltage step
        vset = ps.gen['GEN'].v_setp(x, v)
        if t > 1 and not hasattr(ps, "step_done"):
            for i in range(len(gen_names)):
                vset[i] = 1.1
            ps.step_done = True

        # Measurements
        V = np.abs(v[iB6])

        Pg = ps.gen['GEN'].p_e(x, v) * ps.sys_data['s_n']
        Qg = ps.gen['GEN'].q_e(x, v) * ps.sys_data['s_n']

        Pg1 = Pg[gen_names.index('G1')]
        Qg1 = Qg[gen_names.index('G1')]

        Pg3 = np.nan
        Qg3 = np.nan
        if 'G3' in gen_names:
            Pg3 = Pg[gen_names.index('G3')]
            Qg3 = Qg[gen_names.index('G3')]

        Q_total = np.sum(Qg)
        Q_avg = np.mean(Qg)

        Pvsc, Qvsc = read_vsc(ps, x, v)

        res['t'].append(t)
        res['V'].append(V)
        res['Qg1'].append(Qg1)
        res['Qg3'].append(Qg3)
        res['Q_total'].append(Q_total)
        res['Q_avg'].append(Q_avg)
        res['Qvsc'].append(Qvsc)
        res['Pg1'].append(Pg1)

    return res


# --------------------------------------------------
# MAIN
# --------------------------------------------------
results = {}

for label, P in POWER_CASES.items():
    importlib.reload(model_data)
    base_model = model_data.load()

    model_case = apply_gen_case(base_model, ACTIVE_GENS)
    model_case = distribute_power(model_case, P, ACTIVE_GENS)

    print(f"\nRunning {label} case → {P} MW")

    results[label] = run_simulation(model_case)


# --------------------------------------------------
# PLOTS
# --------------------------------------------------
plt.figure()
for k in results:
    plt.plot(results[k]['t'], results[k]['V'], label=k)
plt.title("Voltage at B6")
plt.xlabel("Time [s]")
plt.ylabel("pu")
plt.legend()


plt.figure()
for k in results:
    plt.plot(results[k]['t'], results[k]['Qg1'], label=k)
plt.title("Reactive Power of G1 ")
plt.xlabel("Time [s]")
plt.ylabel("MVAr")
plt.legend()


plt.figure()
for k in results:
    plt.plot(results[k]['t'], results[k]['Q_avg'], label=k)
plt.title("Average Reactive Power")
plt.xlabel("Time [s]")
plt.ylabel("MVAr")
plt.legend()


plt.figure()
for k in results:
    plt.plot(results[k]['t'], results[k]['Qvsc'], label=k)
plt.title("VSC Reactive Power")
plt.xlabel("Time [s]")
plt.ylabel("MVAr")
plt.legend()




plt.show()