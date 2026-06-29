# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
import importlib
import copy

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
# STUDY SETTINGS
# --------------------------------------------------
LOCAL_UNIT_RATING_MW = 360.0
LOADING_PU = 0.85
BASE_LOCAL_UNIT_MW = LOCAL_UNIT_RATING_MW * LOADING_PU   # 306 MW

FAULT_START = 1.0
SIM_END = 8.0
MAX_STEP = 5e-3
FAULT_Y = 1e6

BUS_IDX = {
    'B1': 0,
    'B2': 1,
    'B5': 2,
    'B6': 3,
    'B7': 4,
    'B8': 5,
    'B9': 6,
    'B10': 7
}

FAULT_BUS = 'B8'
FAULT_BUS_INDEX = BUS_IDX[FAULT_BUS]

GEN_CASES = {
    '1-gen': ['G1', 'G3'],
    '2-gen': ['G1', 'G3', 'G4'],
    '4-gen': ['G1', 'G3', 'G4', 'G5', 'G6'],
}

COARSE_DURATIONS = np.concatenate([
    np.arange(0.05, 1.05, 0.05),
    np.arange(1.10, 2.10, 0.10),
    np.arange(2.20, 3.20, 0.20),
])

REFINE_TOL = 1e-3


def apply_case(model, active_gens):
    model = copy.deepcopy(model)
    keep = set(active_gens)

    header = model['generators']['GEN'][0]
    model['generators']['GEN'] = [header] + [
        row for row in model['generators']['GEN'][1:] if row[0] in keep
    ]

    if 'gov' in model and 'HYGOV' in model['gov']:
        header = model['gov']['HYGOV'][0]
        model['gov']['HYGOV'] = [header] + [
            row for row in model['gov']['HYGOV'][1:] if row[1] in keep
        ]

    if 'avr' in model and 'SEXS' in model['avr']:
        header = model['avr']['SEXS'][0]
        model['avr']['SEXS'] = [header] + [
            row for row in model['avr']['SEXS'][1:] if row[1] in keep
        ]

    if 'pss' in model and 'STAB1' in model['pss']:
        header = model['pss']['STAB1'][0]
        model['pss']['STAB1'] = [header] + [
            row for row in model['pss']['STAB1'][1:] if row[1] in keep
        ]

    return model


def set_local_generation(model, active_gens, p_each=BASE_LOCAL_UNIT_MW):
    model = copy.deepcopy(model)
    header = model['generators']['GEN'][0]
    P_idx = header.index('P')

    for row in model['generators']['GEN'][1:]:
        gname = row[0]
        if gname != 'G1' and gname in active_gens:
            row[P_idx] = p_each

    return model


def scale_t4(model, n_local):
    model = copy.deepcopy(model)

    header = model['transformers'][0]
    S_idx = header.index('S_n')

    for row in model['transformers'][1:]:
        if row[0] == 'T4':
            if n_local == 1:
                row[S_idx] = 360
            elif n_local == 2:
                row[S_idx] = 720
            elif n_local == 4:
                row[S_idx] = 1440

    return model


def read_vsc_q(ps, x, v):
    try:
        if hasattr(ps, 'vsc') and 'VSC_SI' in ps.vsc:
            return float(np.asarray(ps.vsc['VSC_SI'].q_e(x, v)).item()) * ps.sys_data['s_n']
    except:
        pass
    return 0.0


def run_sim(active_gens, fault_duration, record=False):
    importlib.reload(model_data)
    model = model_data.load()

    n_local = len([g for g in active_gens if g != 'G1'])

    model = apply_case(model, active_gens)
    model = set_local_generation(model, active_gens, p_each=BASE_LOCAL_UNIT_MW)
    model = scale_t4(model, n_local)

    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_lib)
    ps.power_flow()
    ps.init_dyn_sim()

    sol = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0,
        ps.x_0.copy(),
        SIM_END,
        max_step=MAX_STEP
    )

    gen_names = list(ps.gen['GEN'].par['name'])
    iG1 = gen_names.index('G1')
    local_names = [g for g in active_gens if g != 'G1']
    local_idx = [gen_names.index(g) for g in local_names]

    res = {
        't': [],
        'delta_group': [],
        'delta_each': {g: [] for g in local_names},
        'Vfault': [],
        'Qg1': [],
        'Qlocal': {g: [] for g in local_names},
        'Qvsc': [],
        'speed_group': [],
    }

    stable = True

    while sol.t < SIM_END:
        t_now = sol.t

        if FAULT_START <= t_now < FAULT_START + fault_duration:
            ps.y_bus_red_mod[(FAULT_BUS_INDEX, FAULT_BUS_INDEX)] = FAULT_Y
        else:
            ps.y_bus_red_mod[(FAULT_BUS_INDEX, FAULT_BUS_INDEX)] = 0

        sol.step()

        t = sol.t
        x = sol.y
        v = sol.v

        angles = np.asarray(ps.gen['GEN'].angle(x, v), dtype=float)
        speeds = np.asarray(ps.gen['GEN'].speed(x, v), dtype=float)
        q_gen = np.asarray(ps.gen['GEN'].q_e(x, v), dtype=float) * ps.sys_data['s_n']

        delta_each = {}
        for g, idx in zip(local_names, local_idx):
            delta_each[g] = float(angles[idx] - angles[iG1])

        delta_group = float(np.mean([delta_each[g] for g in local_names]))
        speed_group = float(np.mean([speeds[idx] for idx in local_idx])) if local_idx else 1.0

        if any(abs(delta_each[g]) > np.pi for g in local_names):
            stable = False
            if record:
                res['t'].append(t)
                res['delta_group'].append(delta_group)
                for g in local_names:
                    res['delta_each'][g].append(delta_each[g])
                res['Vfault'].append(float(np.abs(v[FAULT_BUS_INDEX])))
                res['Qg1'].append(float(q_gen[iG1]))
                for g, idx in zip(local_names, local_idx):
                    res['Qlocal'][g].append(float(q_gen[idx]))
                res['Qvsc'].append(read_vsc_q(ps, x, v))
                res['speed_group'].append(speed_group)
            break

        if record:
            res['t'].append(t)
            res['delta_group'].append(delta_group)
            for g in local_names:
                res['delta_each'][g].append(delta_each[g])
            res['Vfault'].append(float(np.abs(v[FAULT_BUS_INDEX])))
            res['Qg1'].append(float(q_gen[iG1]))
            for g, idx in zip(local_names, local_idx):
                res['Qlocal'][g].append(float(q_gen[idx]))
            res['Qvsc'].append(read_vsc_q(ps, x, v))
            res['speed_group'].append(speed_group)

    return stable, res


def find_cct_for_case(label, active_gens):
    print(f"\nCase: {label} | Active generators: {active_gens}")

    last_stable = None
    first_unstable = None

    for dur in COARSE_DURATIONS:
        stable, _ = run_sim(active_gens, dur, record=False)
        print(f"  fault duration = {dur:.3f} s -> {'STABLE' if stable else 'UNSTABLE'}")

        if stable:
            last_stable = dur
        else:
            first_unstable = dur
            break

    if last_stable is None:
        return {
            'status': 'no stable case found',
            'cct_duration': None,
            'clear_time': None,
            'stable_res': None,
            'unstable_res': None,
        }

    if first_unstable is None:
        return {
            'status': f'no unstable case found up to {last_stable:.3f} s',
            'cct_duration': last_stable,
            'clear_time': FAULT_START + last_stable,
            'stable_res': run_sim(active_gens, last_stable, record=True)[1],
            'unstable_res': None,
        }

    lo = last_stable
    hi = first_unstable

    while (hi - lo) > REFINE_TOL:
        mid = 0.5 * (lo + hi)
        stable, _ = run_sim(active_gens, mid, record=False)
        print(f"    refine {mid:.4f} s -> {'STABLE' if stable else 'UNSTABLE'}")

        if stable:
            lo = mid
        else:
            hi = mid

    cct_duration = lo
    stable_res = run_sim(active_gens, max(0.01, cct_duration - 0.005), record=True)[1]
    unstable_res = run_sim(active_gens, hi, record=True)[1]

    return {
        'status': 'ok',
        'cct_duration': cct_duration,
        'clear_time': FAULT_START + cct_duration,
        'stable_res': stable_res,
        'unstable_res': unstable_res,
    }


def plot_case_result(label, case_result, active_gens):
    stable_res = case_result['stable_res']
    unstable_res = case_result['unstable_res']

    if stable_res is None and unstable_res is None:
        return

    local_names = [g for g in active_gens if g != 'G1']

    plt.figure()
    if stable_res is not None:
        plt.plot(stable_res['t'], stable_res['delta_group'], linewidth=2, label='Stable case')
    if unstable_res is not None:
        plt.plot(unstable_res['t'], unstable_res['delta_group'], '--', linewidth=2, label='Unstable case')
    plt.axhline(np.pi, linestyle=':', color='k', label='±π limit')
    plt.axhline(-np.pi, linestyle=':', color='k')
    plt.title(f"CCT Analysis at Bus B8 ({label}): Average Local Rotor Angle Relative to G1")
    plt.xlabel("Time [s]")
    plt.ylabel("Rotor angle [rad]")
    plt.grid()
    plt.legend()

    plt.figure()
    if stable_res is not None:
        for g in local_names:
            plt.plot(stable_res['t'], stable_res['delta_each'][g], linewidth=2, label=f'{g} stable')
    if unstable_res is not None:
        for g in local_names:
            plt.plot(unstable_res['t'], unstable_res['delta_each'][g], '--', linewidth=2, label=f'{g} unstable')
    plt.axhline(np.pi, linestyle=':', color='k', label='±π limit')
    plt.axhline(-np.pi, linestyle=':', color='k')
    plt.title(f"CCT Analysis at Bus B8 ({label}): Individual Local Generator Angles Relative to G1")
    plt.xlabel("Time [s]")
    plt.ylabel("Rotor angle [rad]")
    plt.grid()
    plt.legend()

    plt.figure()
    if stable_res is not None:
        plt.plot(stable_res['t'], stable_res['Vfault'], linewidth=2, label='Stable case')
    if unstable_res is not None:
        plt.plot(unstable_res['t'], unstable_res['Vfault'], '--', linewidth=2, label='Unstable case')
    plt.title(f"CCT Analysis at Bus B8 ({label}): Voltage Response at Bus B8")
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage magnitude [pu]")
    plt.grid()
    plt.legend()

    plt.figure()
    if stable_res is not None:
        plt.plot(stable_res['t'], stable_res['Qvsc'], linewidth=2, label='Stable case')
    if unstable_res is not None:
        plt.plot(unstable_res['t'], unstable_res['Qvsc'], '--', linewidth=2, label='Unstable case')
    plt.title(f"CCT Analysis at Bus B8 ({label}): VSC Reactive Power Response")
    plt.xlabel("Time [s]")
    plt.ylabel("Reactive power [MVAr]")
    plt.grid()
    plt.legend()

    plt.figure()
    if stable_res is not None:
        plt.plot(stable_res['t'], stable_res['speed_group'], linewidth=2, label='Stable case')
    if unstable_res is not None:
        plt.plot(unstable_res['t'], unstable_res['speed_group'], '--', linewidth=2, label='Unstable case')
    plt.title(f"CCT Analysis at Bus B8 ({label}): Average Local Generator Speed")
    plt.xlabel("Time [s]")
    plt.ylabel("Speed [pu]")
    plt.grid()
    plt.legend()


if __name__ == "__main__":
    print("\n========== CCT SEARCH: FAULT AT B8 ==========")

    all_results = {}

    for label, gens in GEN_CASES.items():
        all_results[label] = find_cct_for_case(label, gens)

    print("\n========== FINAL RESULTS: FAULT AT B8 ==========")
    for label, out in all_results.items():
        print(f"{label}: {out['status']}")
        print(f"  Critical fault duration = {out['cct_duration']}")
        print(f"  Critical clearing instant = {out['clear_time']}")

    for label, gens in GEN_CASES.items():
        plot_case_result(label, all_results[label], gens)

    plt.show()