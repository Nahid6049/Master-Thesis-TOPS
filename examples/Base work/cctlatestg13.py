# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
import importlib

import tops.dynamic as dps
import tops.solvers as dps_sol

# -------------------------------------------------
# PATHS
# -------------------------------------------------
sys.path.append(r"D:\Masters REM+\Master Thesis\paper\TOPS-main\TOPS-main")
sys.path.append(r"D:\Masters REM+\Master Thesis\paper\TOPS-main\TOPS-main\examples\user_models")

import user_lib
import my_network as model_data


# -------------------------------------------------
# SETTINGS
# -------------------------------------------------
T_FAULT = 2.0          # fault starts at 2 s
T_END = 10.0
MAX_STEP = 5e-3
Y_FAULT = 1e6          # strong 3-phase fault at B6
ANGLE_LIMIT = np.pi    # instability threshold


# -------------------------------------------------
# BUILD SYSTEM
# -------------------------------------------------
def build_system():
    importlib.reload(model_data)
    model = model_data.load()

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

    return ps, sol


# -------------------------------------------------
# GET BUS INDEX
# -------------------------------------------------
def get_bus_index():
    return {'B1':0, 'B2':1, 'B5':2, 'B6':3, 'B7':4, 'B8':5, 'B9':6, 'B10':7}


# -------------------------------------------------
# GET ROTOR ANGLE STATE INDICES
# -------------------------------------------------
def get_angle_states(ps):
    angle_states = []
    for i, desc in enumerate(ps.state_desc):
        if len(desc) > 1 and desc[1] == 'angle':
            angle_states.append(i)
    return angle_states


# -------------------------------------------------
# RUN ONE FAULT CASE
# t_clear = FAULT DURATION, not absolute time
# -------------------------------------------------
def run_fault_case(t_clear, return_trace=False):
    ps, sol = build_system()

    idx = get_bus_index()
    iB6 = idx['B6']

    angle_states = get_angle_states(ps)
    if len(angle_states) < 2:
        raise RuntimeError("Need at least 2 generator angle states for relative rotor-angle CCT.")

    time = []
    delta_list = []

    unstable = False

    while sol.t < T_END:
        t = sol.t

        # Apply fault between T_FAULT and T_FAULT + t_clear
        if T_FAULT <= t < T_FAULT + t_clear:
            ps.y_bus_red_mod[iB6, iB6] = Y_FAULT
        else:
            ps.y_bus_red_mod[iB6, iB6] = 0.0

        sol.step()

        x = sol.y
        t = sol.t

        # relative rotor angle
        delta = x[angle_states[1]] - x[angle_states[0]]

        time.append(t)
        delta_list.append(delta)

        if abs(delta) > ANGLE_LIMIT:
            unstable = True
            break

    if return_trace:
        return (not unstable), np.array(time), np.array(delta_list)

    return not unstable


# -------------------------------------------------
# FIND BRACKET FOR CCT
# Finds first unstable duration
# -------------------------------------------------
def find_cct_bracket(start=0.05, step=0.05, max_duration=5.0):
    t_clear = start
    last_stable = 0.0

    while t_clear <= max_duration:
        stable = run_fault_case(t_clear)
        print(f"fault duration = {t_clear:.3f} s -> {'Stable' if stable else 'Unstable'}")

        if stable:
            last_stable = t_clear
            t_clear += step
        else:
            return last_stable, t_clear

    return last_stable, None


# -------------------------------------------------
# COMPUTE CCT
# -------------------------------------------------
def compute_cct(tol=0.005):
    stable_dur, unstable_dur = find_cct_bracket()

    if unstable_dur is None:
        print("\nSystem remained stable for all tested fault durations.")
        print("CCT is greater than the tested maximum duration.")
        return stable_dur, False

    low = stable_dur
    high = unstable_dur

    while (high - low) > tol:
        mid = 0.5 * (low + high)
        stable = run_fault_case(mid)

        print(f"binary search: {mid:.4f} s -> {'Stable' if stable else 'Unstable'}")

        if stable:
            low = mid
        else:
            high = mid

    return low, True


# -------------------------------------------------
# PLOT CCT BOUNDARY
# -------------------------------------------------
def plot_cct_boundary(cct):
    stable1, t1, d1 = run_fault_case(max(cct * 0.95, 0.001), return_trace=True)
    stable2, t2, d2 = run_fault_case(cct * 1.05, return_trace=True)

    clear1 = T_FAULT + max(cct * 0.95, 0.001)
    clear2 = T_FAULT + cct * 1.05
    clear_cct = T_FAULT + cct

    plt.figure(figsize=(9, 6))
    plt.plot(t1, d1, label=f"Stable (~0.95*CCT), stable={stable1}")
    plt.plot(t2, d2, "--", label=f"Unstable (~1.05*CCT), stable={stable2}")
    plt.axvline(clear_cct, linestyle=":", label=f"CCT clearing instant = {clear_cct:.3f} s")

    plt.xlabel("Time [s]")
    plt.ylabel("Relative Rotor Angle [rad]")
    plt.title("Critical Clearing Time (CCT) - Fault at B6")
    plt.grid()
    plt.legend()
    plt.show()


# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":
    cct, bracketed = compute_cct()

    if bracketed:
        print(f"\nEstimated CCT = {cct:.4f} s (fault duration)")
        print(f"Fault starts at {T_FAULT:.3f} s")
        print(f"Clearing instant = {T_FAULT + cct:.4f} s")
        plot_cct_boundary(cct)
    else:
        print(f"\nNo instability found. Tested stable up to fault duration = {cct:.4f} s")