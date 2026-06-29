# simulate_gen_separate.py
# -------------------------------------------------------
# Generator dynamic study (NO PLL)
# Clean thesis-style plots
# -------------------------------------------------------

from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import tops.dynamic as dps
import tops.solvers as dps_sol

import my_network


# -------------------------------------------------------
# MATPLOTLIB STYLE
# -------------------------------------------------------
plt.style.use('default')


if __name__ == "__main__":

    model = my_network.load()

    ps = dps.PowerSystemModel(model=model)

    ps.power_flow()
    ps.init_dyn_sim()

    t_end = 10.0
    x0 = ps.x0.copy()

    sol = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0.0,
        x0,
        t_end,
        max_step=5e-3
    )

    # -------------------------------------------------------
    # Fault at B8
    # -------------------------------------------------------
    bus_names = list(ps.buses["name"])
    fault_bus = "B8"
    k = bus_names.index(fault_bus)

    result_dict = defaultdict(list)

    P_e_stored = []
    E_f_stored = []
    V_stored = []

    t = 0.0
    t0 = time.time()

    # -------------------------------------------------------
    # Simulation loop
    # -------------------------------------------------------
    while t < t_end:

        if 1.0 < t < 1.05:
            ps.y_bus_red_mod[k, k] = 10000
        else:
            ps.y_bus_red_mod[k, k] = 0

        sol.step()

        x = sol.y
        v = sol.v
        t = sol.t

        result_dict[("Global", "t")].append(t)

        for desc, state in zip(ps.state_desc, x):
            result_dict[tuple(desc)].append(state)

        P_e_stored.append(ps.gen["GEN"].P_e(x, v).copy())
        E_f_stored.append(ps.gen["GEN"].E_f(x, v).copy())
        V_stored.append(float(np.abs(v[k])))

    print(f"Simulation completed in {time.time() - t0:.2f} s")

    # -------------------------------------------------------
    # Post-processing
    # -------------------------------------------------------
    result = pd.DataFrame({kk: pd.Series(vv) for kk, vv in result_dict.items()})
    result.columns = pd.MultiIndex.from_tuples(result.columns)

    time_series = result[("Global", "t")].to_numpy()

    speed_df = result.xs(key="speed", axis="columns", level=1)
    angle_df = result.xs(key="angle", axis="columns", level=1)

    P_e = np.array(P_e_stored)
    E_f = np.array(E_f_stored)
    V_mag = np.array(V_stored)

    gen_rows = model["generators"]["GEN"][1:]
    gen_names = [row[0] for row in gen_rows]
    gen_Sn = np.array([row[2] for row in gen_rows], dtype=float)

    P_e_pu = P_e / gen_Sn

    # -------------------------------------------------------
    # Generator Speed
    # -------------------------------------------------------
    fig = plt.figure(figsize=(8,4), facecolor='white')

    plt.plot(time_series, speed_df.to_numpy(), linewidth=2)

    plt.title("Generator Speed Response")
    plt.xlabel("Time [s]")
    plt.ylabel("Speed Deviation [p.u.]")

    plt.legend(gen_names)

    ax = plt.gca()
    ax.set_facecolor('white')

    plt.grid(False)
    plt.tight_layout()

    # -------------------------------------------------------
    # Rotor Angle
    # -------------------------------------------------------
    fig = plt.figure(figsize=(8,4), facecolor='white')

    plt.plot(time_series, angle_df.to_numpy(), linewidth=2)

    plt.title("Generator Rotor Angle Response")
    plt.xlabel("Time [s]")
    plt.ylabel("Rotor Angle [rad]")

    plt.legend(gen_names)

    ax = plt.gca()
    ax.set_facecolor('white')

    plt.grid(False)
    plt.tight_layout()

    # -------------------------------------------------------
    # Electrical Power
    # -------------------------------------------------------
    fig = plt.figure(figsize=(8,4), facecolor='white')

    plt.plot(time_series, P_e_pu, linewidth=2)

    plt.title("Generator Electrical Power Response")
    plt.xlabel("Time [s]")
    plt.ylabel("Electrical Power $P_e$ [p.u.]")

    plt.legend([f"{g} $P_e$" for g in gen_names])

    ax = plt.gca()
    ax.set_facecolor('white')

    plt.grid(False)
    plt.tight_layout()

    # -------------------------------------------------------
    # Field Voltage
    # -------------------------------------------------------
    fig = plt.figure(figsize=(8,4), facecolor='white')

    plt.plot(time_series, E_f, linewidth=2)

    plt.title("Field Voltage Response")
    plt.xlabel("Time [s]")
    plt.ylabel("Field Voltage $E_f$ [p.u.]")

    plt.legend([f"{g} $E_f$" for g in gen_names])

    ax = plt.gca()
    ax.set_facecolor('white')

    plt.grid(False)
    plt.tight_layout()

    # -------------------------------------------------------
    # Voltage at B8
    # -------------------------------------------------------
    fig = plt.figure(figsize=(8,4), facecolor='white')

    plt.plot(time_series, V_mag, linewidth=2)

    plt.title("Voltage Magnitude at Bus B8")
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage Magnitude $|V_{B8}|$ [p.u.]")

    ax = plt.gca()
    ax.set_facecolor('white')

    plt.grid(False)
    plt.tight_layout()

    plt.show()