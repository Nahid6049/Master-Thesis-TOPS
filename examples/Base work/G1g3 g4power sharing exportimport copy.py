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
# USER SETTINGS
# --------------------------------------------------
SIM_TIME = 10.0
MAX_STEP = 2e-3
STEP_TIME = 1.0
LOCAL_VREF_STEP = 1.02   # milder than 1.10

# positive = inject to AC grid
# negative = absorb from AC grid
SCENARIOS = {
    'export': {
        'P_local_total_MW': 600.0,   # total local plant generation
        'VSC_p_ref_pu':  +0.20,      # +200 MW on 1000 MVA base
        'VSC_q_ref_pu':   0.00,
    },
    'balance': {
        'P_local_total_MW': 300.0,
        'VSC_p_ref_pu':   0.00,
        'VSC_q_ref_pu':   0.00,
    },
    'import': {
        'P_local_total_MW': 100.0,
        'VSC_p_ref_pu':  -0.20,      # -200 MW => converter absorbs from AC side
        'VSC_q_ref_pu':   0.00,
    },
}

# 4 topology cases
CASE_DEFS = {
    'Case 1: G1 + G3': {
        'keep_gens': {'G1', 'G3'}
    },
    'Case 2: G1 + G3 + G4': {
        'keep_gens': {'G1', 'G3', 'G4'}
    },
    'Case 3: G1 + G3 + G4 + G5': {
        'keep_gens': {'G1', 'G3', 'G4', 'G5'}
    },
    'Case 4: G1 + G3 + G4 + G5 + G6': {
        'keep_gens': {'G1', 'G3', 'G4', 'G5', 'G6'}
    },
}


# --------------------------------------------------
# MODEL HELPERS
# --------------------------------------------------
def filter_dynamic_rows(rows, keep_names, name_col=0):
    header = rows[0]
    body = [r for r in rows[1:] if r[name_col] in keep_names]
    return [header] + body


def filter_by_gen_column(rows, keep_gens, gen_col=1):
    header = rows[0]
    body = [r for r in rows[1:] if r[gen_col] in keep_gens]
    return [header] + body


def apply_case_topology(model, keep_gens):
    """
    Keep only selected generators and their corresponding gov/avr/pss rows.
    """
    model = copy.deepcopy(model)

    # generators
    model['generators']['GEN'] = filter_dynamic_rows(
        model['generators']['GEN'], keep_gens, name_col=0
    )

    # governors
    model['gov']['HYGOV'] = filter_by_gen_column(
        model['gov']['HYGOV'], keep_gens, gen_col=1
    )

    # avrs
    model['avr']['SEXS'] = filter_by_gen_column(
        model['avr']['SEXS'], keep_gens, gen_col=1
    )

    # pss
    model['pss']['STAB1'] = filter_by_gen_column(
        model['pss']['STAB1'], keep_gens, gen_col=1
    )

    return model


def set_local_generation_dispatch(model, total_local_p_mw):
    """
    Distribute the requested total local active power among all online local machines
    (G3, G4, G5, G6) in proportion to rating S_n.

    G1 is left unchanged as slack-area machine.
    """
    model = copy.deepcopy(model)

    gen_rows = model['generators']['GEN']
    header = gen_rows[0]
    name_idx = header.index('name')
    sn_idx = header.index('S_n')
    p_idx = header.index('P')

    local_names = []
    local_sn = []

    for row in gen_rows[1:]:
        gname = row[name_idx]
        if gname in {'G3', 'G4', 'G5', 'G6'}:
            local_names.append(gname)
            local_sn.append(float(row[sn_idx]))

    total_sn = sum(local_sn)

    if total_sn <= 0 or len(local_names) == 0:
        raise ValueError("No online local generators found among G3-G6.")

    for row in gen_rows[1:]:
        gname = row[name_idx]
        if gname in {'G3', 'G4', 'G5', 'G6'}:
            share = float(row[sn_idx]) / total_sn
            row[p_idx] = total_local_p_mw * share

    return model


def set_vsc_refs(model, p_ref_pu=0.0, q_ref_pu=0.0):
    """
    Update VSC_SI references in the model dictionary.
    """
    model = copy.deepcopy(model)

    vsc_rows = model['vsc']['VSC_SI']
    header = vsc_rows[0]
    p_idx = header.index('p_ref')
    q_idx = header.index('q_ref')

    for row in vsc_rows[1:]:
        row[p_idx] = p_ref_pu
        row[q_idx] = q_ref_pu

    return model


# --------------------------------------------------
# MEASUREMENT HELPERS
# --------------------------------------------------
def sum_selected_gen_power(ps, x, v, selected_names):
    gen_names = list(ps.gen['GEN'].par['name'])
    P_all = ps.gen['GEN'].p_e(x, v) * ps.sys_data['s_n']
    Q_all = ps.gen['GEN'].q_e(x, v) * ps.sys_data['s_n']

    P_sum = 0.0
    Q_sum = 0.0

    for name in selected_names:
        if name in gen_names:
            idx = gen_names.index(name)
            P_sum += P_all[idx]
            Q_sum += Q_all[idx]

    return P_sum, Q_sum


def one_gen_power(ps, x, v, name='G3'):
    gen_names = list(ps.gen['GEN'].par['name'])
    P_all = ps.gen['GEN'].p_e(x, v) * ps.sys_data['s_n']
    Q_all = ps.gen['GEN'].q_e(x, v) * ps.sys_data['s_n']

    if name in gen_names:
        idx = gen_names.index(name)
        return P_all[idx], Q_all[idx]
    return 0.0, 0.0


def get_vsc_power(ps, x, v):
    """
    Reads VSC power from VSC_SI model.
    Assumes one VSC named VSC1.
    """
    if 'VSC_SI' not in ps.vsc:
        return 0.0, 0.0

    Pvsc = ps.vsc['VSC_SI'].p_e(x, v) * ps.sys_data['s_n']
    Qvsc = ps.vsc['VSC_SI'].q_e(x, v) * ps.sys_data['s_n']

    # if only one unit, convert to scalar
    if np.size(Pvsc) == 1:
        Pvsc = float(np.atleast_1d(Pvsc)[0])
        Qvsc = float(np.atleast_1d(Qvsc)[0])

    return Pvsc, Qvsc


# --------------------------------------------------
# SIMULATION
# --------------------------------------------------
def run_simulation(model, local_online_gens):
    """
    Simulate one model instance and log:
    - Bus 6 voltage
    - total local generator P/Q
    - G3 P/Q
    - VSC P/Q
    """

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

    # bus order in your reduced network
    bus_map = {'B1': 0, 'B2': 1, 'B5': 2, 'B6': 3, 'B7': 4, 'B8': 5, 'B9': 6, 'B10': 7}
    iB6 = bus_map['B6']

    res = defaultdict(list)
    t = 0.0
    step_done = False

    while t < SIM_TIME:
        sol.step()

        t = sol.t
        x = sol.y
        v = sol.v

        # --------------------------------------------------
        # mild voltage-reference step for online local generators
        # --------------------------------------------------
        if (t >= STEP_TIME) and (not step_done):
            gen_names = list(ps.gen['GEN'].par['name'])
            try:
                vset = ps.gen['GEN'].v_setp(x, v)
                for g in local_online_gens:
                    if g in gen_names:
                        idx = gen_names.index(g)
                        vset[idx] = LOCAL_VREF_STEP
                step_done = True
            except Exception as e:
                print(f"Warning: could not apply V-step at t={t:.3f}s -> {e}")
                step_done = True

        # --------------------------------------------------
        # measurements
        # --------------------------------------------------
        V_B6 = np.abs(v[iB6])

        P_local, Q_local = sum_selected_gen_power(ps, x, v, local_online_gens)
        P_g3, Q_g3 = one_gen_power(ps, x, v, 'G3')
        P_vsc, Q_vsc = get_vsc_power(ps, x, v)

        res['t'].append(t)
        res['V_B6'].append(V_B6)

        res['P_local'].append(P_local)
        res['Q_local'].append(Q_local)

        res['P_G3'].append(P_g3)
        res['Q_G3'].append(Q_g3)

        res['P_VSC'].append(P_vsc)
        res['Q_VSC'].append(Q_vsc)

        if np.any(np.isnan(x)):
            print("Warning: unstable / NaN encountered. Simulation stopped.")
            break

    return res


# --------------------------------------------------
# PLOTTING
# --------------------------------------------------
def plot_case_results(case_name, case_results):
    labels = list(case_results.keys())

    # 1. Bus 6 voltage
    plt.figure(figsize=(8, 5))
    for lab in labels:
        plt.plot(case_results[lab]['t'], case_results[lab]['V_B6'], label=lab)
    plt.title(f"Voltage at Bus 6 - {case_name}")
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [pu]")
    plt.grid(True)
    plt.legend()

    # 2. total local generator active power
    plt.figure(figsize=(8, 5))
    for lab in labels:
        plt.plot(case_results[lab]['t'], case_results[lab]['P_local'], label=lab)
    plt.title(f"Total Local Generator Active Power - {case_name}")
    plt.xlabel("Time [s]")
    plt.ylabel("MW")
    plt.grid(True)
    plt.legend()

    # 3. total local generator reactive power
    plt.figure(figsize=(8, 5))
    for lab in labels:
        plt.plot(case_results[lab]['t'], case_results[lab]['Q_local'], label=lab)
    plt.title(f"Total Local Generator Reactive Power - {case_name}")
    plt.xlabel("Time [s]")
    plt.ylabel("MVAr")
    plt.grid(True)
    plt.legend()

    # 4. G3 active power
    plt.figure(figsize=(8, 5))
    for lab in labels:
        plt.plot(case_results[lab]['t'], case_results[lab]['P_G3'], label=lab)
    plt.title(f"G3 Active Power - {case_name}")
    plt.xlabel("Time [s]")
    plt.ylabel("MW")
    plt.grid(True)
    plt.legend()

    # 5. G3 reactive power
    plt.figure(figsize=(8, 5))
    for lab in labels:
        plt.plot(case_results[lab]['t'], case_results[lab]['Q_G3'], label=lab)
    plt.title(f"G3 Reactive Power - {case_name}")
    plt.xlabel("Time [s]")
    plt.ylabel("MVAr")
    plt.grid(True)
    plt.legend()

    # 6. VSC active power
    plt.figure(figsize=(8, 5))
    for lab in labels:
        plt.plot(case_results[lab]['t'], case_results[lab]['P_VSC'], label=lab)
    plt.title(f"VSC Active Power - {case_name}")
    plt.xlabel("Time [s]")
    plt.ylabel("MW")
    plt.grid(True)
    plt.legend()

    # 7. VSC reactive power
    plt.figure(figsize=(8, 5))
    for lab in labels:
        plt.plot(case_results[lab]['t'], case_results[lab]['Q_VSC'], label=lab)
    plt.title(f"VSC Reactive Power - {case_name}")
    plt.xlabel("Time [s]")
    plt.ylabel("MVAr")
    plt.grid(True)
    plt.legend()


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    all_results = {}

    for case_name, case_cfg in CASE_DEFS.items():
        print("\n" + "=" * 70)
        print(case_name)
        print("=" * 70)

        keep_gens = case_cfg['keep_gens']
        local_online_gens = sorted([g for g in keep_gens if g in {'G3', 'G4', 'G5', 'G6'}])

        case_results = {}

        for scen_name, scen_cfg in SCENARIOS.items():
            importlib.reload(model_data)
            base_model = model_data.load()

            # topology
            model_case = apply_case_topology(base_model, keep_gens)

            # local dispatch
            model_case = set_local_generation_dispatch(
                model_case,
                total_local_p_mw=scen_cfg['P_local_total_MW']
            )

            # vsc reference
            model_case = set_vsc_refs(
                model_case,
                p_ref_pu=scen_cfg['VSC_p_ref_pu'],
                q_ref_pu=scen_cfg['VSC_q_ref_pu']
            )

            print(f"\nScenario: {scen_name}")
            print(f"  Local online generators: {local_online_gens}")
            print(f"  Total local generation target = {scen_cfg['P_local_total_MW']:.1f} MW")
            print(f"  VSC p_ref = {scen_cfg['VSC_p_ref_pu']:.3f} pu")
            print(f"  VSC q_ref = {scen_cfg['VSC_q_ref_pu']:.3f} pu")

            # show dispatch
            gen_header = model_case['generators']['GEN'][0]
            p_idx = gen_header.index('P')
            for row in model_case['generators']['GEN'][1:]:
                print(f"    {row[0]} -> P0 = {row[p_idx]:.2f} MW")

            # run
            case_results[scen_name] = run_simulation(model_case, local_online_gens)

        all_results[case_name] = case_results
        plot_case_results(case_name, case_results)

    plt.show()
    return all_results


if __name__ == "__main__":
    results = main()