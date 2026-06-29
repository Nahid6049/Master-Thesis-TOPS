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
# SETTINGS: G1 + G3 + G4 + G5 + G6
# --------------------------------------------------
ACTIVE_GENS = ['G1', 'G3', 'G4', 'G5', 'G6']

FAULT_START = 5.0
SIM_END = 12.0
MAX_STEP = 2e-3
FAULT_Y = 10000

DURATION_START = 0.10
DURATION_STOP = 2.0
DURATION_STEP = 0.05
REFINE_TOL = 0.001

BUS_IDX = {
    'B1': 0, 'B2': 1, 'B5': 2, 'B6': 3,
    'B7': 4, 'B8': 5, 'B9': 6, 'B10': 7
}

FAULT_BUS_INDEX = BUS_IDX['B6']


# --------------------------------------------------
# MODEL MODIFICATION
# --------------------------------------------------
def apply_case(model):
    model = copy.deepcopy(model)
    keep = set(ACTIVE_GENS)

    header = model['generators']['GEN'][0]
    model['generators']['GEN'] = [header] + [
        g for g in model['generators']['GEN'][1:] if g[0] in keep
    ]

    for key in ['gov', 'avr', 'pss']:
        subkey = list(model[key].keys())[0]
        header = model[key][subkey][0]
        model[key][subkey] = [header] + [
            g for g in model[key][subkey][1:] if g[1] in keep
        ]

    return model


def set_loading(model):
    model = copy.deepcopy(model)

    header = model['generators']['GEN'][0]
    P_idx = header.index('P')
    S_idx = header.index('S_n')

    for row in model['generators']['GEN'][1:]:
        if row[0] != 'G1':
            row[P_idx] = 0.85 * row[S_idx]

    return model


def scale_t4(model):
    model = copy.deepcopy(model)

    for tr in model['transformers'][1:]:
        if tr[0] == 'T4':
            tr[3] = 1440   # 4 local generators

    return model


# --------------------------------------------------
# SIMULATION
# --------------------------------------------------
def run_sim(fault_duration, record=False):

    importlib.reload(model_data)
    model = model_data.load()

    model = apply_case(model)
    model = set_loading(model)
    model = scale_t4(model)

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
    local_idx = [
        gen_names.index('G3'),
        gen_names.index('G4'),
        gen_names.index('G5'),
        gen_names.index('G6')
    ]

    T_CLEAR = FAULT_START + fault_duration

    stable = True
    max_angle = 0.0

    res = {
        't': [],
        'angle_avg': [],
        'V': [],
        'speed_avg': [],
        'Qvsc': []
    }

    while sol.t < SIM_END:

        t = sol.t

        if 5 < t < T_CLEAR:
            ps.y_bus_red_mod[FAULT_BUS_INDEX, FAULT_BUS_INDEX] = FAULT_Y
        else:
            ps.y_bus_red_mod[FAULT_BUS_INDEX, FAULT_BUS_INDEX] = 0

        sol.step()

        x = sol.y
        v = sol.v

        delta = np.asarray(ps.gen['GEN'].angle(x, v), dtype=float)
        omega = np.asarray(ps.gen['GEN'].speed(x, v), dtype=float)

        local_angles = [delta[i] - delta[iG1] for i in local_idx]
        angle_avg = np.rad2deg(np.mean(local_angles))

        speed_avg = np.mean([omega[i] for i in local_idx])
        V = abs(v[FAULT_BUS_INDEX])

        max_angle = max(max_angle, abs(angle_avg))

        try:
            Qv = ps.vsc['VSC_SI'].q_e(x, v) * ps.sys_data['s_n']
            Qvsc = float(np.asarray(Qv).item())
        except:
            Qvsc = 0.0

        if record:
            res['t'].append(sol.t)
            res['angle_avg'].append(angle_avg)
            res['V'].append(V)
            res['speed_avg'].append(speed_avg)
            res['Qvsc'].append(Qvsc)

        if abs(angle_avg) > 180:
            stable = False
            break

    return stable, res, max_angle


# --------------------------------------------------
# CCT SEARCH
# --------------------------------------------------
def find_cct():

    print("\n===== CCT SEARCH: G1 + G3 + G4 + G5 + G6 =====\n")

    last_stable = None
    first_unstable = None

    durations = np.arange(DURATION_START, DURATION_STOP, DURATION_STEP)

    for d in durations:

        stable, _, max_angle = run_sim(d)

        print(
            f"{d*1000:.0f} ms -> "
            f"{'STABLE' if stable else 'UNSTABLE'} | "
            f"max angle = {max_angle:.2f}"
        )

        if stable:
            last_stable = d
        else:
            first_unstable = d
            break

    if first_unstable is None:
        print("\nNo instability found. Increase DURATION_STOP.")
        return None, None

    lo, hi = last_stable, first_unstable

    print("\nRefining...")

    while hi - lo > REFINE_TOL:
        mid = 0.5 * (lo + hi)

        stable, _, max_angle = run_sim(mid)

        print(
            f"{mid*1000:.2f} ms -> "
            f"{'STABLE' if stable else 'UNSTABLE'} | "
            f"max angle = {max_angle:.2f}"
        )

        if stable:
            lo = mid
        else:
            hi = mid

    print("\n===== FINAL RESULT =====")
    print(f"Last stable  = {lo*1000:.2f} ms")
    print(f"First unstable = {hi*1000:.2f} ms")
    print(f"CCT ≈ {(lo + hi) * 500:.2f} ms")

    stable_res = run_sim(lo, True)[1]
    unstable_res = run_sim(hi, True)[1]

    return stable_res, unstable_res


# --------------------------------------------------
# PLOTS
# --------------------------------------------------
def plot_results(stable, unstable):

    if stable is None or unstable is None:
        return

    plt.figure()
    plt.plot(stable['t'], stable['angle_avg'], label="Last stable")
    plt.plot(unstable['t'], unstable['angle_avg'], '--', label="First unstable")
    plt.axhline(180, linestyle=':')
    plt.axhline(-180, linestyle=':')
    plt.axvline(FAULT_START, linestyle=':')
    plt.title("G1 + G3 + G4 + G5 + G6: Average Local Rotor Angle")
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [deg]")
    plt.grid()
    plt.legend()

    plt.figure()
    plt.plot(stable['t'], stable['V'], label="Last stable")
    plt.plot(unstable['t'], unstable['V'], '--', label="First unstable")
    plt.axvline(FAULT_START, linestyle=':')
    plt.title("G1 + G3 + G4 + G5 + G6: Voltage at B6")
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [pu]")
    plt.grid()
    plt.legend()

    plt.figure()
    plt.plot(stable['t'], stable['speed_avg'], label="Last stable")
    plt.plot(unstable['t'], unstable['speed_avg'], '--', label="First unstable")
    plt.axvline(FAULT_START, linestyle=':')
    plt.title("G1 + G3 + G4 + G5 + G6: Average Local Generator Speed")
    plt.xlabel("Time [s]")
    plt.ylabel("Speed deviation [pu]")
    plt.grid()
    plt.legend()

    plt.figure()
    plt.plot(stable['t'], stable['Qvsc'], label="Last stable")
    plt.plot(unstable['t'], unstable['Qvsc'], '--', label="First unstable")
    plt.axvline(FAULT_START, linestyle=':')
    plt.title("G1 + G3 + G4 + G5 + G6: VSC Reactive Power")
    plt.xlabel("Time [s]")
    plt.ylabel("Reactive Power [MVAr]")
    plt.grid()
    plt.legend()

    plt.show()


# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":

    stable, unstable = find_cct()
    plot_results(stable, unstable)