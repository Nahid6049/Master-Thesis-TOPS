# simulate_pll_angle_vs_voltage.py
# ------------------------------------------------------------
# PLL Deep Study:
# - PLL angle vs actual voltage angle
# - PLL angle error
# - PLL frequency response& "C:\Users\ASUS\AppData\Local\Programs\Python\Python313\python.exe" ".\PLL study\pll_angle_vs_voltage angle.py"
# - Close vs Remote fault comparison
# ------------------------------------------------------------
import sys
import os

# Add Base work directory to Python path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
import numpy as np
import matplotlib.pyplot as plt
import time

import tops.dynamic as dps
import tops.solvers as dps_sol
import my_network


# ------------------------------------------------------------
# Add PLL at selected bus
# ------------------------------------------------------------
def add_pll_at_bus(model, pll_bus="B8", T_filter=0.05):
    model = dict(model)
    model["pll"] = {
        "PLL1": [
            ["name", "T_filter", "bus"],
            ["PLL_" + pll_bus, T_filter, pll_bus],
        ]
    }
    return model


# ------------------------------------------------------------
# Run simulation
# ------------------------------------------------------------
def run_pll_study(
    base_model,
    pll_bus="B8",
    fault_bus="B8",
    t_fault_start=1.0,
    t_fault_end=1.05,
    t_end=10.0,
    fault_admittance=10000.0,
    max_step=5e-3,
):

    model = add_pll_at_bus(base_model, pll_bus=pll_bus)

    ps = dps.PowerSystemModel(model=model)
    ps.power_flow()
    ps.init_dyn_sim()

    x0 = ps.x0.copy()
    sol = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0.0,
        x0,
        t_end,
        max_step=max_step,
    )

    bus_names = list(ps.buses["name"])
    k_pll = bus_names.index(pll_bus)
    k_fault = bus_names.index(fault_bus)

    t_store = []
    V_mag = []
    V_ang = []
    PLL_ang = []
    PLL_freq = []

    t = 0.0
    t0 = time.time()

    while t < t_end:

        # Apply fault
        if t_fault_start < t < t_fault_end:
            ps.y_bus_red_mod[k_fault, k_fault] = fault_admittance
        else:
            ps.y_bus_red_mod[k_fault, k_fault] = 0.0

        sol.step()

        x = sol.y
        v = sol.v
        t = sol.t

        t_store.append(t)

        Vpll = v[k_pll]
        V_mag.append(float(np.abs(Vpll)))
        V_ang.append(float(np.angle(Vpll)))

        f_est = ps.pll["PLL1"].freq_est(x, v)
        theta_est = ps.pll["PLL1"].output(x, v)

        PLL_freq.append(float(np.array(f_est).flatten()[0]))
        PLL_ang.append(float(np.array(theta_est).flatten()[0]))

    print(f"Simulation completed in {time.time() - t0:.2f} s")

    return (
        np.array(t_store),
        np.array(V_mag),
        np.unwrap(np.array(V_ang)),
        np.unwrap(np.array(PLL_ang)),
        np.array(PLL_freq),
    )


# ------------------------------------------------------------
# Plotting function
# ------------------------------------------------------------
def plot_pll_results(
    t, Vmag, Vang, PLLang, PLLfreq,
    pll_bus, fault_bus,
    tfs=1.0, tfe=1.05,
    title_text=""
):

    # 1) Angle comparison
    plt.figure()
    plt.plot(t, PLLang, label="PLL Angle")
    plt.plot(t, Vang, "--", label="Voltage Angle")
    plt.axvspan(tfs, tfe, color="red", alpha=0.2,
                label=f"Fault at {fault_bus}")
    plt.title(f"PLL Angle vs Voltage Angle\n{title_text}")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.grid(True)
    plt.legend()

    # 2) Angle error
    plt.figure()
    plt.plot(t, PLLang - Vang, label="PLL Angle Error")
    plt.axvspan(tfs, tfe, color="red", alpha=0.2)
    plt.title(f"PLL Angle Error\n{title_text}")
    plt.xlabel("Time (s)")
    plt.ylabel("Error (rad)")
    plt.grid(True)
    plt.legend()

    # 3) Frequency (zoom)
    plt.figure()
    mask = (t > (tfs - 0.3)) & (t < (tfe + 1.0))
    plt.plot(t[mask], PLLfreq[mask], label="PLL Frequency")
    plt.axvspan(tfs, tfe, color="red", alpha=0.2)
    plt.title(f"PLL Frequency Response \n{title_text}")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency Deviation")
    plt.grid(True)
    plt.legend()

    # 4) Voltage magnitude
    plt.figure()
    plt.plot(t, Vmag, label="Voltage Magnitude")
    plt.axvspan(tfs, tfe, color="red", alpha=0.2)
    plt.title(f"Voltage Magnitude at {pll_bus}\n{title_text}")
    plt.xlabel("Time (s)")
    plt.ylabel("|V| (p.u.)")
    plt.grid(True)
    plt.legend()


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":

    # =============================
    # Case 1: Close Fault (B8)
    # =============================
    model1 = my_network.load()

    t1, Vmag1, Vang1, PLLang1, PLLfreq1 = run_pll_study(
        model1,
        pll_bus="B8",
        fault_bus="B8",
    )

    plot_pll_results(
        t1, Vmag1, Vang1, PLLang1, PLLfreq1,
        pll_bus="B8",
        fault_bus="B8",
        title_text="PLL at B8 – Fault at B8 (Close Fault)"
    )

    # =============================
    # Case 2: Remote Fault (B2)
    # =============================
    model2 = my_network.load()

    t2, Vmag2, Vang2, PLLang2, PLLfreq2 = run_pll_study(
        model2,
        pll_bus="B8",
        fault_bus="B2",
    )

    plot_pll_results(
        t2, Vmag2, Vang2, PLLang2, PLLfreq2,
        pll_bus="B8",
        fault_bus="B2",
        title_text="PLL at B8 – Fault at B2 (Remote Fault)"
    )

    # =============================
    # Direct Comparison Plot
    # =============================
    plt.figure()
    plt.plot(t1, PLLang1 - Vang1, label="Close Fault (Bus B8)")
    plt.plot(t2, PLLang2 - Vang2, "--", label="Remote Fault (Bus B2)")
    plt.axvspan(1.0, 1.05, color="red", alpha=0.2)
    plt.title("PLL Angle Error Comparison\nPLL Installed at Bus B8")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle Error (rad)")
    plt.grid(True)
    plt.legend()

    plt.show()