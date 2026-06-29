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
# SETTINGS: G1 + G3 ONLY
# --------------------------------------------------
ACTIVE_GENS = ['G1', 'G3']

FAULT_START = 5.0
SIM_END = 12.0
MAX_STEP = 2e-3

FAULT_Y = 10000

# Search settings
DURATION_START = 0.10      # 100 ms
DURATION_STOP = 2.00       # search up to 2000 ms if needed
DURATION_STEP = 0.05       # 50 ms coarse step
REFINE_TOL = 0.001         # 1 ms refinement

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

FAULT_BUS = 'B6'
FAULT_BUS_INDEX = BUS_IDX[FAULT_BUS]


# --------------------------------------------------
# MODEL MODIFICATION
# --------------------------------------------------
def apply_case(model):
    model = copy.deepcopy(model)
    keep = set(ACTIVE_GENS)

    header = model['generators']['GEN'][0]
    model['generators']['GEN'] = [header] + [
        row for row in model['generators']['GEN'][1:]
        if row[0] in keep
    ]

    for key in ['gov', 'avr', 'pss']:
        subkey = list(model[key].keys())[0]
        header = model[key][subkey][0]
        model[key][subkey] = [header] + [
            row for row in model[key][subkey][1:]
            if row[1] in keep
        ]

    return model


def set_loading(model):
    model = copy.deepcopy(model)

    header = model['generators']['GEN'][0]
    P_idx = header.index('P')
    S_idx = header.index('S_n')

    for row in model['generators']['GEN'][1:]:
        if row[0] == 'G3':
            row[P_idx] = 0.85 * row[S_idx]   # 306 MW

    return model


def scale_t4(model):
    model = copy.deepcopy(model)

    for tr in model['transformers'][1:]:
        if tr[0] == 'T4':
            tr[3] = 360

    return model


# --------------------------------------------------
# SINGLE SIMULATION
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
    iG3 = gen_names.index('G3')

    T_CLEAR = FAULT_START + fault_duration

    stable = True
    reason = "stable"

    res = {
        't': [],
        'V_B6': [],
        'angle_G3_G1': [],
        'speed_G3': [],
        'Qvsc': []
    }

    max_angle = 0.0
    max_speed = 0.0
    min_voltage = 999.0

    while sol.t < SIM_END:

        t = sol.t

        # Professor short-circuit block
        if 5 < t < T_CLEAR:
            ps.y_bus_red_mod[FAULT_BUS_INDEX, FAULT_BUS_INDEX] = FAULT_Y
        else:
            ps.y_bus_red_mod[FAULT_BUS_INDEX, FAULT_BUS_INDEX] = 0

        try:
            sol.step()
        except Exception as e:
            stable = False
            reason = f"numerical failure: {e}"
            break

        x = sol.y
        v = sol.v

        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(v)):
            stable = False
            reason = "non-finite state/voltage"
            break

        delta = np.asarray(ps.gen['GEN'].angle(x, v), dtype=float)
        omega = np.asarray(ps.gen['GEN'].speed(x, v), dtype=float)

        angle = np.rad2deg(delta[iG3] - delta[iG1])
        speed = omega[iG3]
        V_B6 = abs(v[FAULT_BUS_INDEX])

        max_angle = max(max_angle, abs(angle))
        max_speed = max(max_speed, abs(speed))
        min_voltage = min(min_voltage, V_B6)

        try:
            Qv = ps.vsc['VSC_SI'].q_e(x, v) * ps.sys_data['s_n']
            Qvsc = float(np.asarray(Qv).item())
        except:
            Qvsc = 0.0

        if record:
            res['t'].append(sol.t)
            res['V_B6'].append(V_B6)
            res['angle_G3_G1'].append(angle)
            res['speed_G3'].append(speed)
            res['Qvsc'].append(Qvsc)

        # Main transient stability criterion
        if abs(angle) > 180:
            stable = False
            reason = "rotor angle exceeded 180 deg"
            if record:
                break
            else:
                break

        # Extra safety criterion: very large speed deviation
        if abs(speed) > 0.20:
            stable = False
            reason = "speed deviation exceeded 0.20 pu"
            if record:
                break
            else:
                break

    return {
        'stable': stable,
        'reason': reason,
        'fault_duration': fault_duration,
        'clearing_time': T_CLEAR,
        'max_angle': max_angle,
        'max_speed': max_speed,
        'min_voltage': min_voltage,
        'res': res
    }


# --------------------------------------------------
# CCT SEARCH
# --------------------------------------------------
def find_cct():

    print("\n===== CCT SEARCH: G1 + G3, FAULT AT B6 =====")
    print("Searching wider until instability is found...\n")

    last_stable = None
    first_unstable = None

    durations = np.arange(DURATION_START, DURATION_STOP + DURATION_STEP, DURATION_STEP)

    for dur in durations:
        out = run_sim(dur, record=False)

        print(
            f"Fault duration {dur*1000:7.1f} ms -> "
            f"{'STABLE' if out['stable'] else 'UNSTABLE'} | "
            f"max angle = {out['max_angle']:7.2f} deg | "
            f"max speed = {out['max_speed']:.4f} pu | "
            f"min V = {out['min_voltage']:.3f} pu"
        )

        if out['stable']:
            last_stable = dur
        else:
            first_unstable = dur
            break

    if last_stable is None:
        print("\nNo stable case found even at the shortest duration.")
        return None, None

    if first_unstable is None:
        print(f"\nNo unstable case found up to {DURATION_STOP*1000:.0f} ms.")
        print("Increase DURATION_STOP if you still need the actual CCT.")
        stable_trace = run_sim(last_stable, record=True)
        return stable_trace, None

    print("\nRefining CCT...")
    lo = last_stable
    hi = first_unstable

    while hi - lo > REFINE_TOL:
        mid = 0.5 * (lo + hi)
        out = run_sim(mid, record=False)

        print(
            f"  test {mid*1000:7.2f} ms -> "
            f"{'STABLE' if out['stable'] else 'UNSTABLE'} | "
            f"max angle = {out['max_angle']:7.2f} deg"
        )

        if out['stable']:
            lo = mid
        else:
            hi = mid

    print("\n===== FINAL CCT RESULT =====")
    print(f"Last stable fault duration  = {lo*1000:.2f} ms")
    print(f"First unstable duration     = {hi*1000:.2f} ms")
    print(f"Estimated CCT               = {(lo+hi)*500:.2f} ms")
    print(f"Fault applied at            = {FAULT_START:.3f} s")
    print(f"Critical clearing instant   = {FAULT_START + lo:.3f} to {FAULT_START + hi:.3f} s")

    stable_trace = run_sim(lo, record=True)
    unstable_trace = run_sim(hi, record=True)

    return stable_trace, unstable_trace


# --------------------------------------------------
# PLOTS
# --------------------------------------------------
def plot_results(stable_trace, unstable_trace):

    traces = []

    if stable_trace is not None:
        traces.append(("Last stable", stable_trace, "-"))

    if unstable_trace is not None:
        traces.append(("First unstable", unstable_trace, "--"))

    plt.figure()
    for name, out, style in traces:
        r = out['res']
        plt.plot(r['t'], r['V_B6'], style, label=f"{name}: {out['fault_duration']*1000:.1f} ms")
    plt.axvline(FAULT_START, linestyle=':', label='Fault applied')
    plt.title("G1 + G3: Voltage at B6")
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [pu]")
    plt.grid()
    plt.legend()

    plt.figure()
    for name, out, style in traces:
        r = out['res']
        plt.plot(r['t'], r['angle_G3_G1'], style, label=f"{name}: {out['fault_duration']*1000:.1f} ms")
    plt.axvline(FAULT_START, linestyle=':', label='Fault applied')
    plt.axhline(180, linestyle=':', label='+180 deg')
    plt.axhline(-180, linestyle=':', label='-180 deg')
    plt.title("G1 + G3: Rotor Angle Difference G3 - G1")
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [deg]")
    plt.grid()
    plt.legend()

    plt.figure()
    for name, out, style in traces:
        r = out['res']
        plt.plot(r['t'], r['speed_G3'], style, label=f"{name}: {out['fault_duration']*1000:.1f} ms")
    plt.axvline(FAULT_START, linestyle=':', label='Fault applied')
    plt.title("G1 + G3: Generator Speed of G3")
    plt.xlabel("Time [s]")
    plt.ylabel("Speed deviation [pu]")
    plt.grid()
    plt.legend()

    plt.figure()
    for name, out, style in traces:
        r = out['res']
        plt.plot(r['t'], r['Qvsc'], style, label=f"{name}: {out['fault_duration']*1000:.1f} ms")
    plt.axvline(FAULT_START, linestyle=':', label='Fault applied')
    plt.title("G1 + G3: VSC Reactive Power")
    plt.xlabel("Time [s]")
    plt.ylabel("Reactive Power [MVAr]")
    plt.grid()
    plt.legend()

    plt.show()


# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":

    stable_trace, unstable_trace = find_cct()
    plot_results(stable_trace, unstable_trace)