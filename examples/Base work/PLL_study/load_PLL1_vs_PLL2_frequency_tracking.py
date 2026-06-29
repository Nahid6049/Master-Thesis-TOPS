# PLL1_vs_PLL2_with_metrics.py
# --------------------------------------------
# Compare PLL1 vs PLL2 under load step
# Includes automatic metrics table
# --------------------------------------------
import tops
print("TOPS loaded from:", tops.__path__)
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

sys.path.append(r"D:\Masters REM+\Master Thesis\thesis work\TOPS-main\TOPS-main")

import tops.dynamic as dps
import tops.solvers as dps_sol
import my_network
import user_lib


# ============================
# SETTINGS
# ============================
PLL_BUS = "B8"
T_END = 20.0
T_EVENT = 10.0
MAX_STEP = 5e-3
LOAD_STEP_FACTOR = 0.10
LOAD_BUSES = ["B5", "B6"]


def wrap_to_pi(x):
    return (x + np.pi) % (2*np.pi) - np.pi


def attach_pll(model, pll_type):

    model = dict(model)

    if pll_type == "PLL1":
        model["pll"] = {
            "PLL1": [
                ["name", "T_filter", "bus"],
                ["PLL_B8", 0.05, PLL_BUS],
            ]
        }

    elif pll_type == "PLL2":
        model["pll"] = {
            "PLL2": [
                ["name", "bus", "K_p", "K_i"],
                ["PLL_B8", PLL_BUS, 5.0, 100.0],
            ]
        }

    return model


def get_reduced_idx(ps, full_idx):
    for attr in ("k_bus_red", "bus_red_map", "bus_reduction_idx", "k_red"):
        if hasattr(ps, attr):
            m = getattr(ps, attr)
            try:
                return int(m[full_idx])
            except:
                pass
    return full_idx


def get_Zload_S_pu_by_bus(model):
    base_mva = float(model.get("base_mva", 1000.0))
    loads = model["loads"]

    header = loads[0]
    i_bus = header.index("bus")
    i_P = header.index("P")
    i_Q = header.index("Q")
    i_model = header.index("model")

    Sbus = {}

    for row in loads[1:]:
        if row[i_model].upper() == "Z":
            bus = row[i_bus]
            P = float(row[i_P]) / base_mva
            Q = float(row[i_Q]) / base_mva
            Sbus[bus] = Sbus.get(bus, 0) + (P + 1j*Q)

    return Sbus


def run_case(pll_type):

    model = my_network.load()
    model = attach_pll(model, pll_type)
    Sbus = get_Zload_S_pu_by_bus(model)

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

    load_red_idx = []
    y_diag_orig = {}

    for b in LOAD_BUSES:
        k_full = bus_names.index(b)
        k_red = get_reduced_idx(ps, k_full)
        load_red_idx.append((b, k_full, k_red))
        y_diag_orig[b] = ps.y_bus_red_mod[k_red, k_red].copy()

    t_store = []
    v_ang_store = []
    pll_ang_store = []

    stepped = False
    y_added = {}

    while sol.t < T_END:

        if (not stepped) and sol.t >= T_EVENT:

            v_now = sol.v

            for b, k_full, k_red in load_red_idx:

                if b not in Sbus:
                    continue

                Vmag = np.abs(v_now[k_full])
                if Vmag < 1e-6:
                    Vmag = 1e-6

                S = Sbus[b]
                Y_base = np.conj(S) / (Vmag**2)
                dY = LOAD_STEP_FACTOR * Y_base

                y_added[b] = dY
                ps.y_bus_red_mod[k_red, k_red] = y_diag_orig[b] + dY

            stepped = True

        if stepped:
            for b, _, k_red in load_red_idx:
                ps.y_bus_red_mod[k_red, k_red] = y_diag_orig[b] + y_added[b]

        sol.step()

        x = sol.y
        v = sol.v

        t_store.append(sol.t)

        Vpll = v[k_pll]
        v_ang_store.append(np.angle(Vpll))

        theta_est = ps.pll[pll_type].output(x, v)
        pll_ang_store.append(float(np.array(theta_est).flatten()[0]))

    t = np.array(t_store)
    v_ang = np.array(v_ang_store)
    pll_ang = np.array(pll_ang_store)

    angle_error = wrap_to_pi(pll_ang - v_ang)

    # Metrics after disturbance
    idx = t >= T_EVENT
    max_err = np.max(np.abs(angle_error[idx]))
    rms_err = np.sqrt(np.mean(angle_error[idx]**2))
    final_err = angle_error[-1]

    return t, angle_error, max_err, rms_err, final_err


# ============================
# RUN BOTH
# ============================
if __name__ == "__main__":

    print("\nRunning PLL1...")
    t1, err1, max1, rms1, final1 = run_case("PLL1")

    print("Running PLL2...")
    t2, err2, max2, rms2, final2 = run_case("PLL2")

    # Plot
    plt.figure()
    plt.plot(t1, err1, label="PLL1")
    plt.plot(t2, err2, "--", label="PLL2")
    plt.axvline(T_EVENT, color="r", linestyle="--")
    plt.title("Angle Error: PLL1 vs PLL2 (Load Step)")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle Error (rad)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Table
    print("\n==========================================")
    print("        PLL Performance Comparison")
    print("==========================================")
    print(f"{'Metric':<25}{'PLL1':<15}{'PLL2':<15}")
    print("------------------------------------------")
    print(f"{'Max |Angle Error|':<25}{max1:<15.6f}{max2:<15.6f}")
    print(f"{'RMS Angle Error':<25}{rms1:<15.6f}{rms2:<15.6f}")
    print(f"{'Final Error @20s':<25}{final1:<15.6f}{final2:<15.6f}")
    print("==========================================")