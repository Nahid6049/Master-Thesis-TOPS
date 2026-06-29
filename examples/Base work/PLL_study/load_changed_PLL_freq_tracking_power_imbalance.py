import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Path setup
# ----------------------------
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import tops.dynamic as dps
import tops.solvers as dps_sol
import my_network
import user_lib

# ============================
# SETTINGS
# ============================
PLL_BUS = "B8"
PLL_CLASS = "PLL1"       # "PLL1" or "PLL2"

T_END = 20.0
T_EVENT = 10.0
MAX_STEP = 5e-3

LOAD_STEP_FACTOR = 0.10
LOAD_BUSES = ["B5", "B6"]
F_NOM = 50.0  # Hz


def wrap_to_pi(x):
    return (x + np.pi) % (2*np.pi) - np.pi


def attach_pll(model):
    model = dict(model)

    if PLL_CLASS == "PLL1":
        model["pll"] = {
            "PLL1": [
                ["name", "T_filter", "bus"],
                [f"PLL_{PLL_BUS}", 0.05, PLL_BUS],
            ]
        }
    elif PLL_CLASS == "PLL2":
        model["pll"] = {
            "PLL2": [
                ["name", "bus", "K_p", "K_i"],
                [f"PLL_{PLL_BUS}", PLL_BUS, 5.0, 100.0],
            ]
        }
    else:
        raise ValueError("PLL_CLASS must be 'PLL1' or 'PLL2'")

    return model


def run_sim():

    model = my_network.load()
    model = attach_pll(model)

    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_lib)
    ps.power_flow()
    ps.init_dyn_sim()

    sol = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0.0,
        ps.x0.copy(),
        T_END,
        max_step=MAX_STEP,
    )

    bus_names = list(ps.buses["name"])
    k_pll = bus_names.index(PLL_BUS)

    # --- Find generator speed states ---
    speed_idx = []
    for i, desc in enumerate(ps.state_desc):
        if len(desc) >= 2 and desc[1] == "speed":
            speed_idx.append(i)

    t_store = []
    v_ang_store = []
    pll_ang_store = []
    pll_freq_store = []
    speed_store = []

    stepped = False

    while sol.t < T_END:

        # Load disturbance
        if (not stepped) and sol.t >= T_EVENT:
            print(f"Applying load step at t = {sol.t:.6f}s")
            stepped = True

        sol.step()

        x = sol.y
        v = sol.v
        t = sol.t

        t_store.append(t)

        # Generator speed
        if speed_idx:
            speed_store.append([x[i] for i in speed_idx])

        # Voltage angle at PLL bus
        v_ang_store.append(float(np.angle(v[k_pll])))

        # PLL outputs
        theta_est = ps.pll[PLL_CLASS].output(x, v)
        pll_ang_store.append(float(np.array(theta_est).flatten()[0]))

        f_est = ps.pll[PLL_CLASS].freq_est(x, v)
        pll_freq_store.append(float(np.array(f_est).flatten()[0]))

    # Convert arrays
    t = np.array(t_store)
    v_ang = np.array(v_ang_store)
    pll_ang = np.array(pll_ang_store)
    pll_freq = np.array(pll_freq_store)
    speed = np.array(speed_store)

    return t, v_ang, pll_ang, pll_freq, speed


if __name__ == "__main__":

    print("Running PLL frequency tracking study")
    print("PLL:", PLL_CLASS)

    t, v_ang, pll_ang, pll_freq, speed = run_sim()

    # ============================
    # ANGLE ANALYSIS
    # ============================

    v_ang_rel = wrap_to_pi(v_ang - v_ang[0])
    pll_ang_rel = wrap_to_pi(pll_ang - pll_ang[0])
    ang_err = wrap_to_pi(pll_ang - v_ang)

    # ============================
    # SYSTEM FREQUENCY (from generators)
    # ============================

    omega_avg = np.mean(speed, axis=1)      # per-unit frequency
    f_sys = omega_avg * F_NOM              # Hz
    f_pll = pll_freq * F_NOM               # Hz
    freq_err = f_pll - f_sys               # Hz error

    # ============================
    # METRICS (after disturbance)
    # ============================

    idx = t >= T_EVENT

    print("\n========== FREQUENCY METRICS ==========")
    print("Max |f_PLL - f_sys| (Hz):", np.max(np.abs(freq_err[idx])))
    print("RMS freq error (Hz):", np.sqrt(np.mean(freq_err[idx]**2)))
    print("Final freq error (Hz):", freq_err[-1])

    print("\n========== ANGLE METRICS ==========")
    print("Max |angle error| (rad):", np.max(np.abs(ang_err[idx])))
    print("RMS angle error (rad):", np.sqrt(np.mean(ang_err[idx]**2)))
    print("Final angle error (rad):", ang_err[-1])

    # ============================
    # PLOTS
    # ============================

    # Relative angles
    plt.figure()
    plt.plot(t, v_ang_rel, label="Voltage angle (relative)")
    plt.plot(t, pll_ang_rel, "--", label="PLL angle (relative)")
    plt.axvline(T_EVENT, color="r", linestyle="--")
    plt.title("PLL Angle Tracking")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.grid(True)
    plt.legend()

    # Angle error
    plt.figure()
    plt.plot(t, ang_err)
    plt.axvline(T_EVENT, color="r", linestyle="--")
    plt.title("Angle Tracking Error")
    plt.xlabel("Time (s)")
    plt.ylabel("Error (rad)")
    plt.grid(True)

    # Frequency comparison
    plt.figure()
    plt.plot(t, f_sys, label="System frequency (Hz)")
    plt.plot(t, f_pll, "--", label="PLL frequency (Hz)")
    plt.axvline(T_EVENT, color="r", linestyle="--")
    plt.title("Frequency Tracking")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.grid(True)
    plt.legend()

    # Frequency error
    plt.figure()
    plt.plot(t, freq_err)
    plt.axvline(T_EVENT, color="r", linestyle="--")
    plt.title("Frequency Tracking Error")
    plt.xlabel("Time (s)")
    plt.ylabel("Error (Hz)")
    plt.grid(True)

    plt.show()