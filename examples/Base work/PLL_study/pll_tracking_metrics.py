import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time

# Make Base work visible
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import tops.dynamic as dps
import tops.solvers as dps_sol
import my_network


# ============================
# Global Parameters
# ============================

PLL_BUS = "B8"
T_FAULT_START = 1.0
T_FAULT_END = 1.05
T_END = 10.0
FAULT_ADMITTANCE = 10000.0


# ============================
# Simulation Function
# ============================

def run_case(fault_bus):

    model = my_network.load()

    model["pll"] = {
        "PLL1": [
            ["name", "T_filter", "bus"],
            ["PLL_AT_B8", 0.05, PLL_BUS],
        ]
    }

    ps = dps.PowerSystemModel(model=model)
    ps.power_flow()
    ps.init_dyn_sim()

    x0 = ps.x0.copy()

    sol = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0.0,
        x0,
        T_END,
        max_step=5e-3,
    )

    bus_names = list(ps.buses["name"])
    k_pll = bus_names.index(PLL_BUS)
    k_fault = bus_names.index(fault_bus)

    t_store = []
    V_mag = []
    V_ang = []
    PLL_ang = []
    PLL_freq = []

    t = 0.0

    while t < T_END:

        if T_FAULT_START < t < T_FAULT_END:
            ps.y_bus_red_mod[k_fault, k_fault] = FAULT_ADMITTANCE
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

    # Convert
    t_store = np.array(t_store)
    V_mag = np.array(V_mag)
    V_ang = np.unwrap(np.array(V_ang))
    PLL_ang = np.unwrap(np.array(PLL_ang))
    PLL_freq = np.array(PLL_freq)

    angle_error = PLL_ang - V_ang

    # Metrics
    min_voltage = np.min(V_mag)
    max_angle_error = np.max(np.abs(angle_error))
    rms_angle_error = np.sqrt(np.mean(angle_error**2))
    max_freq_dev = np.max(np.abs(PLL_freq))

    # Settling time
    settling_time = np.nan
    post_fault_idx = np.where(t_store >= T_FAULT_END)[0]

    if len(post_fault_idx) > 0:
        epost = np.abs(angle_error[post_fault_idx])
        for i in range(len(epost)):
            if np.all(epost[i:] < 0.05):
                settling_time = t_store[post_fault_idx[i]] - T_FAULT_END
                break

    return {
        "fault_bus": fault_bus,
        "t": t_store,
        "angle_error": angle_error,
        "min_voltage": min_voltage,
        "max_angle_error": max_angle_error,
        "rms_angle_error": rms_angle_error,
        "max_freq_dev": max_freq_dev,
        "settling_time": settling_time,
    }


# ============================
# Run Both Cases
# ============================

print("\nRunning Close Fault (B8)...")
close_case = run_case("B8")

print("Running Remote Fault (B2)...")
remote_case = run_case("B2")


# ============================
# Print Comparison
# ============================

print("\n========= COMPARISON =========")

for case in [close_case, remote_case]:
    print(f"\nFault at {case['fault_bus']}")
    print(f"Minimum Voltage: {case['min_voltage']:.4f} p.u.")
    print(f"Max Angle Error: {case['max_angle_error']:.4f} rad")
    print(f"RMS Angle Error: {case['rms_angle_error']:.4f} rad")
    print(f"Max Frequency Deviation: {case['max_freq_dev']:.4f}")
    print(f"Settling Time: {case['settling_time']:.4f} s")

print("\n================================")


# ============================
# Plot Comparison
# ============================

plt.figure()
plt.plot(close_case["t"], close_case["angle_error"], label="Fault at B8 (Close)")
plt.plot(remote_case["t"], remote_case["angle_error"], "--", label="Fault at B2 (Remote)")
plt.axvspan(T_FAULT_START, T_FAULT_END, color="red", alpha=0.3)
plt.title("PLL Angle Error Comparison")
plt.xlabel("Time (s)")
plt.ylabel("Error (rad)")
plt.grid()
plt.legend()

plt.show()