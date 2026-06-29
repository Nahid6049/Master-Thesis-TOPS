# -*- coding: utf-8 -*-

import sys
import copy
import importlib
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import tops.dynamic as dps
import tops.solvers as dps_sol

BASE_PATH = r"D:\Masters REM+\Master Thesis\paper\TOPS-main\TOPS-main"

sys.path.insert(0, BASE_PATH + r"\examples\Base work")
sys.path.append(BASE_PATH + r"\examples\user_models")

import user_lib
import generator_network as model_data


FAULT_BUS = 3
FAULT_START = 5.0
FAULT_Y = 10000.0

T_END = 9.0
MAX_STEP = 5e-3
POST_UNSTABLE_TIME = 0.5

LOCAL_UNIT_RATING = 360
LOADING_PU = 0.85
P_LOCAL_UNIT = LOCAL_UNIT_RATING * LOADING_PU

ANGLE_LIMIT = 180
SCAN_STEP = 0.01
TOL = 1e-4

GEN_CASES = {
    "G1 only": ['G1'],
    "G1 + G3": ['G1', 'G3'],
    "G1 + G3 + G4": ['G1', 'G3', 'G4'],
    "G1 + G3 + G4 + G5 + G6": ['G1', 'G3', 'G4', 'G5', 'G6']
}


def compute_scr(model, n_local):

    Z_L56 = Z_L25 = Z_L69 = None
    Z_T1 = Z_T4 = None
    Z_G1 = None
    Z_G3 = None

    S_base = model["base_mva"]

    for line in model["lines"][1:]:
        Z = line[7] + 1j * line[8]

        if line[0] == "L5-6":
            Z_L56 = Z
        elif line[0] == "L2-5":
            Z_L25 = Z
        elif line[0] == "L6-9":
            Z_L69 = Z

    for tr in model["transformers"][1:]:
        Z = tr[6] + 1j * tr[7]

        if tr[0] == "T1":
            Z_T1 = Z
        elif tr[0] == "T4":
            Z_T4 = Z

    for gen in model["generators"]["GEN"][1:]:
        if gen[0] == "G1":
            Z_G1 = 1j * gen[12]
        elif gen[0] == "G3":
            Z_G3 = 1j * gen[12]

    Z_grid = Z_L25 + Z_L56 + Z_T1 + Z_G1

    if n_local == 0 or Z_G3 is None:
        Z_th = Z_grid
    else:
        # Professor's correction:
        # Zlocal = j0.31 is treated on 360 MVA base,
        # then converted to 1000 MVA base.
        Z_local_old = Z_L69 + Z_T4 + Z_G3
        Z_local_new = Z_local_old * (S_base / LOCAL_UNIT_RATING)

        Z_local_eq = Z_local_new / n_local

        Z_th = 1 / ((1 / Z_grid) + (1 / Z_local_eq))

    SCR = 1 / abs(Z_th)

    return SCR


def prepare_model(active_gens):

    importlib.reload(model_data)
    model = copy.deepcopy(model_data.load())

    h = model['generators']['GEN'][0]
    model['generators']['GEN'] = [h] + [
        g for g in model['generators']['GEN'][1:] if g[0] in active_gens
    ]

    for key in ['gov', 'avr', 'pss']:
        sub = list(model[key].keys())[0]
        h = model[key][sub][0]
        model[key][sub] = [h] + [
            row for row in model[key][sub][1:] if row[1] in active_gens
        ]

    local_gens = [g for g in active_gens if g != 'G1']

    h = model['generators']['GEN'][0]
    P_idx = h.index('P')

    for g in model['generators']['GEN'][1:]:
        if g[0] in local_gens:
            g[P_idx] = P_LOCAL_UNIT

    for tr in model['transformers'][1:]:
        if tr[0] == 'T4':
            if len(local_gens) > 0:
                tr[3] = LOCAL_UNIT_RATING * len(local_gens)

    return model


def read_angle_speed(ps, x, v):

    gen = ps.gen['GEN']

    if hasattr(gen, 'angle'):
        angle = gen.angle(x, v)
    elif hasattr(gen, 'delta'):
        angle = gen.delta(x, v)
    else:
        raise AttributeError("Cannot find rotor angle method.")

    if hasattr(gen, 'speed'):
        speed = gen.speed(x, v)
    elif hasattr(gen, 'omega'):
        speed = gen.omega(x, v)
    else:
        raise AttributeError("Cannot find speed method.")

    return np.asarray(angle), np.asarray(speed)


def read_vsc_Q(ps, x, v):

    try:
        q = ps.vsc['VSC_SI'].q_e(x, v) * ps.sys_data['s_n']
        return float(np.asarray(q).flatten()[0])
    except Exception:
        return 0.0


def run_fault(model, fault_duration, save=False):

    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_lib)

    ps.power_flow()
    ps.init_dyn_sim()

    sol = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0,
        ps.x_0.copy(),
        T_END,
        max_step=MAX_STEP
    )

    clear_time = FAULT_START + fault_duration

    gen_names = list(ps.gen['GEN'].par['name'])

    i_g1 = gen_names.index('G1')
    i_local = [i for i, g in enumerate(gen_names) if g != 'G1']

    angle0, speed0 = read_angle_speed(ps, sol.y, sol.v)
    angle_g1_0 = angle0[i_g1]

    res = defaultdict(list)
    stable = True
    unstable_time = None

    while sol.t < T_END:

        if FAULT_START < sol.t < clear_time:
            ps.y_bus_red_mod[FAULT_BUS, FAULT_BUS] = FAULT_Y
        else:
            ps.y_bus_red_mod[FAULT_BUS, FAULT_BUS] = 0.0

        sol.step()

        x = sol.y
        v = sol.v

        angle, speed = read_angle_speed(ps, x, v)

        if len(i_local) > 0:
            angle_rel = np.rad2deg(np.mean(angle[i_local]) - angle[i_g1])
            speed_avg = np.mean(speed[i_local])
        else:
            angle_rel = np.rad2deg(angle[i_g1] - angle_g1_0)
            speed_avg = speed[i_g1]

        V_B6 = abs(v[FAULT_BUS])
        Q_vsc = read_vsc_Q(ps, x, v)

        if abs(angle_rel) > ANGLE_LIMIT:
            stable = False

            if unstable_time is None:
                unstable_time = sol.t

            if not save:
                break

        if save:
            res['t'].append(sol.t)
            res['angle_rel'].append(angle_rel)
            res['speed_avg'].append(speed_avg)
            res['V_B6'].append(V_B6)
            res['Q_vsc'].append(Q_vsc)

            if unstable_time is not None:
                if sol.t > unstable_time + POST_UNSTABLE_TIME:
                    break

    if save:
        return stable, res

    return stable


def find_cct(model):

    d = 0.0
    last_stable = 0.0
    first_unstable = None

    print("Automatic scan:")

    while d <= 1.0:
        stable = run_fault(model, d)

        print(f"  {d:.3f} s -> {'STABLE' if stable else 'UNSTABLE'}")

        if stable:
            last_stable = d
            d += SCAN_STEP
        else:
            first_unstable = d
            break

    if first_unstable is None:
        return None

    low = last_stable
    high = first_unstable

    print("Binary search:")

    while high - low > TOL:
        mid = (low + high) / 2
        stable = run_fault(model, mid)

        print(f"  {mid:.5f} s -> {'STABLE' if stable else 'UNSTABLE'}")

        if stable:
            low = mid
        else:
            high = mid

    return low


models = {}
cct_results = {}
scr_results = {}

for case_name, active_gens in GEN_CASES.items():

    print("\n==================================================")
    print(f"CASE: {case_name}")
    print("==================================================")

    model = prepare_model(active_gens)
    models[case_name] = model

    n_local = len([g for g in active_gens if g != 'G1'])

    SCR = compute_scr(model, n_local)
    scr_results[case_name] = SCR

    print(f"SCR ≈ {SCR:.2f}")
    print(f"Local loading = {LOADING_PU} pu per active local unit")
    print(f"Power per local unit = {P_LOCAL_UNIT:.2f} MW")
    print(f"Total local power = {P_LOCAL_UNIT * n_local:.2f} MW")

    if n_local > 0:
        print(f"T4 rating = {LOCAL_UNIT_RATING * n_local:.0f} MVA")
    else:
        print("T4 rating = unchanged base value")

    cct = find_cct(model)
    cct_results[case_name] = cct

    if cct is not None:
        print(f"CCT = {cct:.5f} s = {cct * 1000:.2f} ms")
    else:
        print("CCT not found within scan range.")


print("\n================ FINAL RESULTS ================")
print(f"{'Case':35s} {'SCR':10s} {'CCT [s]':12s} {'CCT [ms]':12s}")
print("--------------------------------------------------------------------")

for case_name in GEN_CASES:
    SCR = scr_results[case_name]
    cct = cct_results[case_name]

    if cct is not None:
        print(f"{case_name:35s} {SCR:10.2f} {cct:12.5f} {cct * 1000:12.2f}")
    else:
        print(f"{case_name:35s} {SCR:10.2f} {'Not found':12s} {'Not found':12s}")


MARGIN = 0.005

for case_name, model in models.items():

    cct = cct_results[case_name]

    if cct is None:
          continue

    stable_d = max(cct - MARGIN, 0.0)
    unstable_d = cct + MARGIN

    stable, ds = run_fault(model, stable_d, save=True)
    unstable, du = run_fault(model, unstable_d, save=True)

    # Rotor angle
    plt.figure()
    plt.plot(ds['t'], ds['angle_rel'], label="Stable")
    plt.plot(du['t'], du['angle_rel'], label="Unstable")
    plt.axvline(FAULT_START, linestyle='--', label='Fault Start')
    plt.axvline(FAULT_START + cct, linestyle=':', label='CCT')
    plt.title(f"Relative Rotor Angle (SCR = {scr_results[case_name]:.1f})")
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [deg]")
    plt.legend()
    plt.grid(False)

    # Voltage
    plt.figure()
    plt.plot(ds['t'], ds['V_B6'], label="Stable")
    plt.plot(du['t'], du['V_B6'], label="Unstable")
    plt.axvline(FAULT_START, linestyle='--', label='Fault Start')
    plt.axvline(FAULT_START + cct, linestyle=':', label='CCT')
    plt.title(f"Voltage at Bus B6 (SCR = {scr_results[case_name]:.1f})")
    plt.xlabel("Time [s]")
    plt.ylabel("pu")
    plt.legend()
    plt.grid(False)

    # VSC reactive power
    plt.figure()
    plt.plot(ds['t'], ds['Q_vsc'], label="Stable")
    plt.plot(du['t'], du['Q_vsc'], label="Unstable")
    plt.axvline(FAULT_START, linestyle='--', label='Fault Start')
    plt.axvline(FAULT_START + cct, linestyle=':', label='CCT')
    plt.title(f"VSC Reactive Power (SCR = {scr_results[case_name]:.1f})")
    plt.xlabel("Time [s]")
    plt.ylabel("MVAr")
    plt.legend()
    plt.grid(False)

    # Generator speed
    plt.figure()
    plt.plot(ds['t'], ds['speed_avg'], label="Stable")
    plt.plot(du['t'], du['speed_avg'], label="Unstable")
    plt.axvline(FAULT_START, linestyle='--', label='Fault Start')
    plt.axvline(FAULT_START + cct, linestyle=':', label='CCT')
    plt.title(f"Generator Speed (SCR = {scr_results[case_name]:.1f})")
    plt.xlabel("Time [s]")
    plt.ylabel("pu")
    plt.legend()
    plt.grid(False)

plt.show()