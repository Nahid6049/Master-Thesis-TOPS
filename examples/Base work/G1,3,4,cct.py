import sys
import numpy as np
import matplotlib.pyplot as plt
import importlib
import copy

import tops.dynamic as dps
import tops.solvers as dps_sol

# ---------------- PATH ----------------
BASE_PATH = r"D:\Masters REM+\Master Thesis\paper\TOPS-main\TOPS-main"

sys.path.insert(0, BASE_PATH + r"\examples\Base work")
sys.path.append(BASE_PATH + r"\examples\user_models")

import user_lib
import generator_network as model_data

# ---------------- USER SETTINGS ----------------
FAULT_BUS = 'B6'          # try 'B10' if B6 stays very stable
T_FAULT = 1.0             # fault starts at 1.0 s
T_END = 4.0               # shorter runtime
MAX_STEP = 3e-3
LOADING_PU = 0.85
FAULT_ADMITTANCE = 1e6    # solid 3-phase bus fault in Ybus form

CASES = {
    "G1+G3": ['G1', 'G3'],
    "G1+G3+G4": ['G1', 'G3', 'G4'],
    "G1+G3+G4+G5+G6": ['G1', 'G3', 'G4', 'G5', 'G6'],
}

# coarse/fine search on FAULT DURATION, not absolute clearing time
COARSE_DURS = np.arange(0.05, 1.05, 0.05)
FINE_STEP = 0.005


# ---------------- MODEL PREP ----------------
def prepare_model(active_gens):
    importlib.reload(model_data)
    model = copy.deepcopy(model_data.load())

    def filt(block, idx):
        header = block[0]
        return [header] + [row for row in block[1:] if row[idx] in active_gens]

    model['generators']['GEN'] = filt(model['generators']['GEN'], 0)
    model['gov']['HYGOV'] = filt(model['gov']['HYGOV'], 1)
    model['avr']['SEXS'] = filt(model['avr']['SEXS'], 1)
    model['pss']['STAB1'] = filt(model['pss']['STAB1'], 1)

    # same pu loading for all online local generators
    header = model['generators']['GEN'][0]
    P_idx = header.index('P')
    S_idx = header.index('S_n')

    for row in model['generators']['GEN'][1:]:
        if row[0] != 'G1':
            row[P_idx] = LOADING_PU * row[S_idx]

    return model


# ---------------- ONE DYNAMIC RUN ----------------
def simulate_case(active_gens, fault_duration):
    model = prepare_model(active_gens)

    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_lib)
    ps.power_flow()
    ps.init_dyn_sim()

    bus_names = list(ps.buses['name'])
    fault_idx = bus_names.index(FAULT_BUS)

    gen_names = list(ps.gen['GEN'].par['name'])
    ref_idx = gen_names.index('G1') if 'G1' in gen_names else 0

    sol = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0,
        ps.x_0.copy(),
        T_END,
        max_step=MAX_STEP
    )

    fault_clear_time = T_FAULT + fault_duration

    t_list = []
    spread_list = []
    rel_angle_list = []
    v_faultbus_list = []

    unstable = False

    while sol.t < T_END:
        # Reset network EVERY step first
        ps.y_bus_red_mod = ps.y_bus_red.copy()

        # Apply fault only during fault interval
        if T_FAULT <= sol.t < fault_clear_time:
            ps.y_bus_red_mod[fault_idx, fault_idx] += FAULT_ADMITTANCE

        sol.step()

        t = sol.t
        x = sol.y
        v = sol.v

        angles = np.asarray(ps.gen['GEN'].angle(x, v)).astype(float)
        rel_angles = angles - angles[ref_idx]
        spread = np.max(angles) - np.min(angles)
        V_faultbus = np.abs(v[fault_idx])

        t_list.append(t)
        spread_list.append(spread)
        rel_angle_list.append(rel_angles.copy())
        v_faultbus_list.append(V_faultbus)

        # standard transient stability criterion
        if spread > np.pi:
            unstable = True
            break

    return {
        't': np.array(t_list),
        'spread': np.array(spread_list),
        'rel_angles': np.array(rel_angle_list),
        'V_faultbus': np.array(v_faultbus_list),
        'gen_names': gen_names,
        'stable': not unstable,
        'fault_duration': fault_duration,
    }


# ---------------- CCT SEARCH ----------------
def find_cct(active_gens):
    # guaranteed stable reference
    last_stable = simulate_case(active_gens, 0.0)
    last_stable_dur = 0.0
    first_unstable = None

    # coarse search
    for dur in COARSE_DURS:
        res = simulate_case(active_gens, dur)
        print(f"[coarse] fault_duration={dur:.3f} s -> {'STABLE' if res['stable'] else 'UNSTABLE'}")

        if res['stable']:
            last_stable = res
            last_stable_dur = dur
        else:
            first_unstable = res
            break

    # no unstable point found in coarse range
    if first_unstable is None:
        return {
            'cct': last_stable_dur,
            'stable_case': last_stable,
            'unstable_case': None,
            'found_unstable': False
        }

    # fine search
    fine_start = last_stable_dur
    fine_end = first_unstable['fault_duration']

    stable_case = last_stable
    unstable_case = first_unstable
    cct = fine_start

    for dur in np.arange(fine_start + FINE_STEP, fine_end + 1e-12, FINE_STEP):
        res = simulate_case(active_gens, dur)
        print(f"[fine]   fault_duration={dur:.3f} s -> {'STABLE' if res['stable'] else 'UNSTABLE'}")

        if res['stable']:
            stable_case = res
            cct = dur
        else:
            unstable_case = res
            break

    return {
        'cct': cct,
        'stable_case': stable_case,
        'unstable_case': unstable_case,
        'found_unstable': True
    }


# ---------------- PLOTTING ----------------
def plot_case(case_name, result):
    stable_case = result['stable_case']
    unstable_case = result['unstable_case']

    # 1) Rotor angle spread
    plt.figure()
    plt.plot(
        stable_case['t'], stable_case['spread'],
        linewidth=2, label=f"Stable ({stable_case['fault_duration']:.3f} s)"
    )

    if unstable_case is not None:
        plt.plot(
            unstable_case['t'], unstable_case['spread'],
            '--', linewidth=2, label=f"Unstable ({unstable_case['fault_duration']:.3f} s)"
        )

    plt.axhline(np.pi, linestyle=':', label='π limit')
    plt.title(f"{case_name} - Rotor Angle Spread")
    plt.xlabel("Time [s]")
    plt.ylabel("δmax - δmin [rad]")
    plt.legend()
    plt.grid()

    # 2) Relative rotor angles in same figure
    plt.figure()
    gen_names = stable_case['gen_names']

    for i, g in enumerate(gen_names):
        if g == 'G1':
            continue
        plt.plot(
            stable_case['t'], stable_case['rel_angles'][:, i],
            linewidth=2, label=f"{g} stable"
        )

    if unstable_case is not None:
        for i, g in enumerate(unstable_case['gen_names']):
            if g == 'G1':
                continue
            plt.plot(
                unstable_case['t'], unstable_case['rel_angles'][:, i],
                '--', linewidth=2, label=f"{g} unstable"
            )

    plt.title(f"{case_name} - Relative Rotor Angles (ref = G1)")
    plt.xlabel("Time [s]")
    plt.ylabel("δi - δG1 [rad]")
    plt.legend()
    plt.grid()

    # 3) Voltage at faulted bus
    plt.figure()
    plt.plot(
        stable_case['t'], stable_case['V_faultbus'],
        linewidth=2, label=f"Stable ({stable_case['fault_duration']:.3f} s)"
    )

    if unstable_case is not None:
        plt.plot(
            unstable_case['t'], unstable_case['V_faultbus'],
            '--', linewidth=2, label=f"Unstable ({unstable_case['fault_duration']:.3f} s)"
        )

    plt.title(f"{case_name} - Voltage at {FAULT_BUS}")
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [pu]")
    plt.legend()
    plt.grid()


# ---------------- MAIN ----------------
all_results = {}

for case_name, active_gens in CASES.items():
    print("\n==================================================")
    print(f"CASE: {case_name} | Active generators: {active_gens}")
    print("==================================================")

    result = find_cct(active_gens)
    all_results[case_name] = result

    if result['found_unstable']:
        print(f"\n>>> CCT for {case_name} = {result['cct']:.3f} s")
        print(f"    Stable fault duration used for plot   = {result['stable_case']['fault_duration']:.3f} s")
        print(f"    Unstable fault duration used for plot = {result['unstable_case']['fault_duration']:.3f} s")
    else:
        print(f"\n>>> No unstable case found for {case_name} up to {COARSE_DURS[-1]:.3f} s")
        print(f"    Last tested stable duration = {result['cct']:.3f} s")
        print("    Try FAULT_BUS = 'B10' or extend COARSE_DURS.")

    plot_case(case_name, result)

plt.show()