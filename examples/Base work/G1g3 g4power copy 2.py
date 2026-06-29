# -*- coding: utf-8 -*-

import sys
import copy
import importlib
import numpy as np
import matplotlib.pyplot as plt
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
# SETTINGS
# --------------------------------------------------
SIM_TIME = 10.0
MAX_STEP = 2e-3
STEP_TIME = 1.0
LOCAL_VREF_STEP = 1.02

# --------------------------------------------------
# SCENARIOS (CHANGE TOTAL POWER)
# --------------------------------------------------
SCENARIOS = {
    'export': 600.0,
    'balance': 300.0,
    'import': 100.0,
}

# --------------------------------------------------
# CASES
# --------------------------------------------------
CASES = {
    'Case 1': {'G1','G3'},
    'Case 2': {'G1','G3','G4'},
    #'Case 3': {'G1','G3','G4','G5'},
    'Case 4': {'G1','G3','G4','G5','G6'},
}

# --------------------------------------------------
# APPLY TOPOLOGY
# --------------------------------------------------
def apply_case(model, keep):

    model = copy.deepcopy(model)

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
# SET GENERATION (DISTRIBUTION)
# --------------------------------------------------
def set_generation(model, total_power):

    model = copy.deepcopy(model)

    header = model['generators']['GEN'][0]
    P_idx = header.index('P')

    locals = [g for g in model['generators']['GEN'][1:] if g[0] != 'G1']
    share = total_power / len(locals)

    for g in locals:
        g[P_idx] = share

    return model

# --------------------------------------------------
# SIMULATION
# --------------------------------------------------
def run(model, local_gens):

    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_lib)

    ps.power_flow()
    ps.init_dyn_sim()

    sol = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0,
        ps.x_0.copy(),
        SIM_TIME,
        max_step=MAX_STEP
    )

    bus_map = {'B1':0,'B2':1,'B5':2,'B6':3,'B7':4,'B8':5,'B9':6,'B10':7}
    iB6 = bus_map['B6']

    res = defaultdict(list)
    step_done = False

    while sol.t < SIM_TIME:

        sol.step()

        t = sol.t
        x = sol.y
        v = sol.v

        # voltage step
        if (t >= STEP_TIME) and (not step_done):

            gen_names = list(ps.gen['GEN'].par['name'])
            vset = ps.gen['GEN'].v_setp(x, v)

            for g in local_gens:
                if g in gen_names:
                    idx = gen_names.index(g)
                    vset[idx] = LOCAL_VREF_STEP

            step_done = True

        V = np.abs(v[iB6])

        P = ps.gen['GEN'].p_e(x, v)*ps.sys_data['s_n']
        Q = ps.gen['GEN'].q_e(x, v)*ps.sys_data['s_n']

        # total local
        P_local = 0.0
        Q_local = 0.0

        gen_names = list(ps.gen['GEN'].par['name'])

        for g in local_gens:
            if g in gen_names:
                idx = gen_names.index(g)
                P_local += P[idx]
                Q_local += Q[idx]

        # G3 only
        if 'G3' in gen_names:
            idx_g3 = gen_names.index('G3')
            P_g3 = P[idx_g3]
            Q_g3 = Q[idx_g3]
        else:
            P_g3 = 0.0
            Q_g3 = 0.0

        # VSC
        Pvsc = ps.vsc['VSC_SI'].p_e(x, v)*ps.sys_data['s_n']
        Qvsc = ps.vsc['VSC_SI'].q_e(x, v)*ps.sys_data['s_n']

        res['t'].append(t)
        res['V'].append(V)
        res['P_local'].append(P_local)
        res['Q_local'].append(Q_local)
        res['P_g3'].append(P_g3)
        res['Q_g3'].append(Q_g3)
        res['Pvsc'].append(float(np.atleast_1d(Pvsc)[0]))
        res['Qvsc'].append(float(np.atleast_1d(Qvsc)[0]))

    return res

# --------------------------------------------------
# MAIN
# --------------------------------------------------
for case_name, gens in CASES.items():

    print(f"\n==== {case_name} ====")

    local_gens = [g for g in gens if g != 'G1']
    results = {}

    for scen, Pval in SCENARIOS.items():

        importlib.reload(model_data)
        base = model_data.load()

        model = apply_case(base, gens)
        model = set_generation(model, Pval)

        print(f"Running {scen} | Total={Pval}")

        results[scen] = run(model, local_gens)

    # ------------------ PLOTS ------------------

    # Voltage
    plt.figure()
    for k in results:
        plt.plot(results[k]['t'], results[k]['V'], label=k)
    plt.title(f"Voltage - {case_name}")
    plt.legend()
    plt.grid()

    # TOTAL vs G3 ACTIVE
    plt.figure()
    for k in results:
        plt.plot(results[k]['t'], results[k]['P_local'], label=f"{k} TOTAL")
        plt.plot(results[k]['t'], results[k]['P_g3'], '--', label=f"{k} G3")
    plt.title(f"Total vs G3 Active Power - {case_name}")
    plt.legend()
    plt.grid()

    # TOTAL vs G3 REACTIVE
    plt.figure()
    for k in results:
        plt.plot(results[k]['t'], results[k]['Q_local'], label=f"{k} TOTAL")
        plt.plot(results[k]['t'], results[k]['Q_g3'], '--', label=f"{k} G3")
    plt.title(f"Total vs G3 Reactive Power - {case_name}")
    plt.legend()
    plt.grid()

    # VSC
    plt.figure()
    for k in results:
        plt.plot(results[k]['t'], results[k]['Pvsc'], label=f"{k} P")
    plt.title(f"VSC Active Power - {case_name}")
    plt.legend()
    plt.grid()

    plt.figure()
    for k in results:
        plt.plot(results[k]['t'], results[k]['Qvsc'], label=f"{k} Q")
    plt.title(f"VSC Reactive Power - {case_name}")
    plt.legend()
    plt.grid()

    plt.show()
    plt.close('all')