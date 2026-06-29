# -*- coding: utf-8 -*-

import sys
import copy
import importlib
import numpy as np
import matplotlib.pyplot as plt

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
FAULT_BUS_NAME = 'B6'
FAULT_BUS_INDEX = 3          # B6 = 3
FAULT_START = 5.0            # fault starts at 5 s
FAULT_ADMITTANCE = 10000     # solid 3-phase fault approximation
SIM_END = 10.0
MAX_STEP = 2e-3

# Stability criterion:
# unstable if any local machine angle relative to G1 exceeds pi rad
ANGLE_LIMIT = np.pi

# Cases to test
CLEARING_TIMES = np.arange(0.05, 0.40, 0.01)   # 50 ms to 390 ms


# --------------------------------------------------
# OPTIONAL: CHOOSE GENERATOR CASE
# --------------------------------------------------
ACTIVE_GENS = ['G1', 'G3', 'G4', 'G5', 'G6']   # modify if needed


# --------------------------------------------------
# MODEL PREPARATION
# --------------------------------------------------
def apply_gen_case(model, active_gens):
    model = copy.deepcopy(model)

    # Generators
    header = model['generators']['GEN'][0]
    model['generators']['GEN'] = [header] + [
        g for g in model['generators']['GEN'][1:] if g[0] in active_gens
    ]

    # Gov / AVR / PSS
    for key in ['gov', 'avr', 'pss']:
        subkey = list(model[key].keys())[0]
        header = model[key][subkey][0]
        model[key][subkey] = [header] + [
            row for row in model[key][subkey][1:] if row[1] in active_gens
        ]

    return model


def distribute_local_power(model, total_local_p, active_gens):
    """
    Distribute total active power among local machines (everything except G1).
    Example:
        total_local_p = 300  -> balance
        total_local_p = 600  -> export
        total_local_p = 100  -> import
    """
    model = copy.deepcopy(model)

    local_gens = [g for g in active_gens if g != 'G1']
    if len(local_gens) == 0:
        return model

    share = total_local_p / len(local_gens)

    header = model['generators']['GEN'][0]
    P_idx = header.index('P')

    for row in model['generators']['GEN'][1:]:
        if row[0] in local_gens:
            row[P_idx] = share

    return model


# --------------------------------------------------
# ANGLE READING
# --------------------------------------------------
def get_generator_angles(ps, x, gen_names):
    """
    Read rotor angles from TOPS generator model.
    Depending on model structure, this may be:
      ps.gen['GEN'].angle(x, v)
      or ps.gen['GEN'].delta(x, v)
      or accessible through state descriptions.
    """
    gen = ps.gen['GEN']

    # Try common method names
    for fn_name in ['angle', 'delta']:
        if hasattr(gen, fn_name):
            fn = getattr(gen, fn_name)
            try:
                vals = fn(x, ps.v_0 if hasattr(ps, 'v_0') else None)
                return np.asarray(vals).flatten()
            except:
                try:
                    vals = fn(x, None)
                    return np.asarray(vals).flatten()
                except:
                    pass

    # Fallback using state description
    if hasattr(gen, 'state_idx_global') and hasattr(gen, 'state_list'):
        angles = []
        for g in gen_names:
            found = False
            for state_name in ['angle', 'delta']:
                key = (g, state_name)
                if key in gen.state_idx_global:
                    angles.append(x[gen.state_idx_global[key]])
                    found = True
                    break
            if not found:
                raise RuntimeError(f"Could not find rotor angle state for generator {g}.")
        return np.asarray(angles)

    raise RuntimeError("Could not access generator rotor angles in this TOPS model.")


# --------------------------------------------------
# SINGLE SIMULATION
# --------------------------------------------------
def run_fault_case(model, clearing_time, fault_start=FAULT_START, sim_end=SIM_END):
    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_lib)

    ps.power_flow()
    ps.init_dyn_sim()

    sol = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0,
        ps.x_0.copy(),
        sim_end,
        max_step=MAX_STEP
    )

    gen_names = list(ps.gen['GEN'].par['name'])

    if 'G1' not in gen_names:
        raise ValueError("G1 must be online because stability is checked relative to G1.")

    local_gens = [g for g in gen_names if g != 'G1']

    results = {
        't': [],
        'angles': {g: [] for g in gen_names},
        'delta_rel': {g: [] for g in local_gens},
        'stable': True,
        'clearing_time': clearing_time
    }

    unstable = False

    while sol.t < sim_end:
        sol.step()

        t = sol.t
        x = sol.y
        v = sol.v

        # --------------------------------------------------
        # APPLY / CLEAR FAULT AT B6
        # --------------------------------------------------
        if fault_start <= t < clearing_time:
            ps.y_bus_red_mod[FAULT_BUS_INDEX, FAULT_BUS_INDEX] = FAULT_ADMITTANCE
        else:
            ps.y_bus_red_mod[FAULT_BUS_INDEX, FAULT_BUS_INDEX] = 0

        # --------------------------------------------------
        # READ ANGLES
        # --------------------------------------------------
        ang = get_generator_angles(ps, x, gen_names)

        g1_idx = gen_names.index('G1')
        ang_g1 = ang[g1_idx]

        for i, g in enumerate(gen_names):
            results['angles'][g].append(ang[i])

        for g in local_gens:
            gi = gen_names.index(g)
            rel = ang[gi] - ang_g1

            # wrap to [-pi, pi] only for display if desired
            # but for instability detection, raw separation can also be used
            results['delta_rel'][g].append(rel)

            if abs(rel) > ANGLE_LIMIT:
                unstable = True

        results['t'].append(t)

        # stop early if clearly unstable
        if unstable and t > clearing_time + 0.1:
            results['stable'] = False
            break

    if not unstable:
        results['stable'] = True

    return results


# --------------------------------------------------
# CCT SEARCH
# --------------------------------------------------
def search_cct(model, clearing_times):
    stable_times = []
    unstable_times = []

    first_stable_case = None
    first_unstable_case = None

    for tc in clearing_times:
        actual_clear = FAULT_START + tc
        print(f"Testing fault duration = {tc:.3f} s  -> clearing at t = {actual_clear:.3f} s")

        res = run_fault_case(model, clearing_time=actual_clear)

        if res['stable']:
            print("   Stable")
            stable_times.append(tc)
            first_stable_case = res
        else:
            print("   Unstable")
            unstable_times.append(tc)
            if first_unstable_case is None:
                first_unstable_case = res

    cct = max(stable_times) if stable_times else None
    return cct, stable_times, unstable_times, first_stable_case, first_unstable_case


# --------------------------------------------------
# PLOT STABLE VS UNSTABLE
# --------------------------------------------------
def plot_case_comparison(stable_case, unstable_case):
    plt.figure(figsize=(10, 6))

    if stable_case is not None:
        for g, vals in stable_case['delta_rel'].items():
            plt.plot(
                stable_case['t'], vals,
                label=f"Stable: {g} - G1 (tc={stable_case['clearing_time']-FAULT_START:.3f}s)"
            )

    if unstable_case is not None:
        for g, vals in unstable_case['delta_rel'].items():
            plt.plot(
                unstable_case['t'], vals, '--',
                label=f"Unstable: {g} - G1 (tc={unstable_case['clearing_time']-FAULT_START:.3f}s)"
            )

    plt.axhline(np.pi, color='k', linestyle=':', linewidth=1)
    plt.axhline(-np.pi, color='k', linestyle=':', linewidth=1)
    plt.xlabel("Time [s]")
    plt.ylabel("Rotor angle difference [rad]")
    plt.title("Rotor angle stability comparison for fault at B6")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    importlib.reload(model_data)
    base_model = model_data.load()

    # choose generator set
    model = apply_gen_case(base_model, ACTIVE_GENS)

    # choose operating point if needed
    # export = 600, balance = 300, import = 100
    model = distribute_local_power(model, total_local_p=300, active_gens=ACTIVE_GENS)

    cct, stable_times, unstable_times, stable_case, unstable_case = search_cct(model, CLEARING_TIMES)

    print("\n==============================")
    print("CCT SEARCH RESULT")
    print("==============================")
    if cct is not None:
        print(f"Estimated CCT ≈ {cct:.3f} s fault duration")
        print(f"Fault start = {FAULT_START:.3f} s")
        print(f"Critical clearing time ≈ {FAULT_START + cct:.3f} s absolute time")
    else:
        print("No stable case found in tested range.")

    print(f"Stable durations tested: {stable_times}")
    print(f"Unstable durations tested: {unstable_times}")

    plot_case_comparison(stable_case, unstable_case)