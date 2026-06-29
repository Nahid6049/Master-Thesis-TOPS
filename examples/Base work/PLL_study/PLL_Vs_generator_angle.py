import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Make Base work visible
# ----------------------------
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

# ----------------------------
# Make TOPS root visible (for user_lib)
# ----------------------------
sys.path.append(r"D:\Masters REM+\Master Thesis\thesis work\TOPS-main\TOPS-main")

import tops.dynamic as dps
import tops.solvers as dps_sol
import my_network
import user_lib


# ============================
# SETTINGS
# ============================
PLL_BUS = "B8"
FAULT_BUS = "B8"       # Change to "B2" for remote fault
T_FAULT_START = 1.0
T_FAULT_END = 1.05
T_END = 10.0
FAULT_ADMITTANCE = 10000.0
T_FILTER = 0.05
MAX_STEP = 5e-3


# ----------------------------
# Helper: reduced Y-bus index
# ----------------------------
def _get_reduced_idx(ps, full_idx):
    for attr in ("k_bus_red", "bus_red_map", "bus_reduction_idx", "k_red"):
        if hasattr(ps, attr):
            try:
                return int(getattr(ps, attr)[full_idx])
            except Exception:
                pass
    return full_idx


# ============================
# MAIN
# ============================
if __name__ == "__main__":

    model = my_network.load()

    # Add PLL block
    model["pll"] = {
        "PLL1": [
            ["name", "T_filter", "bus"],
            [f"PLL_{PLL_BUS}", float(T_FILTER), PLL_BUS],
        ]
    }

    # IMPORTANT: pass user_lib
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

    # Bus indices
    bus_names = list(ps.buses["name"])
    k_pll_full = bus_names.index(PLL_BUS)
    k_fault_full = bus_names.index(FAULT_BUS)
    k_fault_red = _get_reduced_idx(ps, k_fault_full)

    # Store original Y-bus diagonal
    y_diag_orig = ps.y_bus_red_mod[k_fault_red, k_fault_red].copy()

    # Identify generator rotor angles
    gen_angle_idx = []
    gen_names = []

    for i, desc in enumerate(ps.state_desc):
        if desc[1] == "angle":
            gen_angle_idx.append(i)
            gen_names.append(desc[0])

    # Storage
    t_store = []
    pll_angle = []
    voltage_angle = []
    gen_angles = [[] for _ in gen_angle_idx]

    # ----------------------------
    # Simulation loop
    # ----------------------------
    while sol.t < T_END:

        # Apply fault
        if T_FAULT_START < sol.t < T_FAULT_END:
            ps.y_bus_red_mod[k_fault_red, k_fault_red] = y_diag_orig + FAULT_ADMITTANCE
        else:
            ps.y_bus_red_mod[k_fault_red, k_fault_red] = y_diag_orig

        sol.step()

        x = sol.y
        v = sol.v
        t = sol.t

        t_store.append(t)

        # Generator rotor angles
        for j, idx in enumerate(gen_angle_idx):
            gen_angles[j].append(x[idx])

        # Voltage angle at PLL bus
        Vpll = v[k_pll_full]
        voltage_angle.append(np.angle(Vpll))

        # PLL angle
        theta_est = ps.pll["PLL1"].output(x, v)
        pll_angle.append(float(np.array(theta_est).flatten()[0]))

    # ----------------------------
    # Convert to arrays
    # ----------------------------
    t_store = np.array(t_store)
    voltage_angle = np.unwrap(np.array(voltage_angle))
    pll_angle = np.unwrap(np.array(pll_angle))
    gen_angles = [np.unwrap(np.array(g)) for g in gen_angles]

    # ----------------------------
    # Plot
    # ----------------------------
    plt.figure(figsize=(10,6))

    for i, g in enumerate(gen_angles):
        plt.plot(t_store, g, label=f"{gen_names[i]} rotor angle")

    plt.plot(t_store, voltage_angle, "--", label=f"Voltage angle ({PLL_BUS})")
    plt.plot(t_store, pll_angle, linewidth=2, label="PLL angle")

    plt.axvspan(T_FAULT_START, T_FAULT_END, color="red", alpha=0.25)

    plt.title("PLL Angle vs Generator Rotor Angles")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.grid(True)
    plt.legend()
    plt.show()