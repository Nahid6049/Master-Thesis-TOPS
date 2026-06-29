# ================================
# CASE: G1 + G3 + G4 + G5 + G6
# ================================

import sys
import numpy as np
import matplotlib.pyplot as plt
import importlib
import copy
from collections import defaultdict

import tops.dynamic as dps
import tops.solvers as dps_sol

BASE_PATH = r"D:\Masters REM+\Master Thesis\paper\TOPS-main\TOPS-main"

sys.path.insert(0, BASE_PATH + r"\examples\Base work")
sys.path.append(BASE_PATH + r"\examples\user_models")

import user_lib
import generator_network as model_data


ACTIVE_GENS = ['G1', 'G3', 'G4', 'G5', 'G6']
TITLE = " Case: G1 + G3 + G4 + G5 + G6"

POWER_CASES = {
    'export': 300,
    'balance': 200,
    'import': 100
}


def apply_gen_case(model):
    model = copy.deepcopy(model)

    def filter_block(block, idx):
        header = block[0]
        return [header] + [row for row in block[1:] if row[idx] in ACTIVE_GENS]

    model['generators']['GEN'] = filter_block(model['generators']['GEN'], 0)
    model['gov']['HYGOV'] = filter_block(model['gov']['HYGOV'], 1)
    model['avr']['SEXS'] = filter_block(model['avr']['SEXS'], 1)
    model['pss']['STAB1'] = filter_block(model['pss']['STAB1'], 1)

    return model


def apply_power(model, total_P):
    model = copy.deepcopy(model)

    local_gens = [g for g in ACTIVE_GENS if g != 'G1']
    share = total_P / len(local_gens)

    header = model['generators']['GEN'][0]
    P_idx = header.index('P')

    for row in model['generators']['GEN'][1:]:
        if row[0] in local_gens:
            row[P_idx] = share

    return model


def run(model):
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

    idx = {'B6': 3}
    iB6 = idx['B6']

    res = defaultdict(list)

    while sol.t < 10:
        sol.step()

        x, v = sol.y, sol.v
        gen_names = list(ps.gen['GEN'].par['name'])

        if sol.t > 1 and not hasattr(ps, "step_done"):
            vset = ps.gen['GEN'].v_setp(x, v)
            for i in range(len(vset)):
                vset[i] = 1.02
            ps.step_done = True

        V = np.abs(v[iB6])
        Q = ps.gen['GEN'].q_e(x, v) * ps.sys_data['s_n']

        res['t'].append(sol.t)
        res['V'].append(float(V))
        res['Qavg'].append(float(np.mean(Q)))

        # Track G3
        res['Qg3'].append(float(Q[gen_names.index('G3')]))

        try:
            vsc = ps.vsc['VSC_SI']
            Qvsc = vsc.q_e(x, v) * vsc.par['S_n']
            res['Qvsc'].append(float(np.asarray(Qvsc).flatten()[0]))
        except Exception:
            res['Qvsc'].append(0.0)

    return res


# MAIN
results = {}

for k, P in POWER_CASES.items():

    importlib.reload(model_data)
    m = model_data.load()

    m = apply_gen_case(m)
    m = apply_power(m, P)

    print(f"{k} → total local generation = {P} MW")

    results[k] = run(m)


# PLOTS
for key, ylabel, title in [
    ('V', 'pu', 'Voltage at B6'),
    ('Qg3', 'MVAr', 'Reactive Power of G3'),
    ('Qvsc', 'MVAr', 'VSC Reactive Power')
]:
    plt.figure()

    for k in results:
        plt.plot(
            results[k]['t'],
            results[k][key],
            label=k
        )

    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel(ylabel)
    plt.legend()
    

plt.show()