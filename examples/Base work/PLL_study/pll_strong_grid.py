import sys
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TOPS_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))

if TOPS_ROOT not in sys.path:
    sys.path.insert(0, TOPS_ROOT)

import user_lib
import my_network




if __name__ == "__main__":

    # --- Load network ---
    model = my_network.load()

    # --- Add PLL at B8 (measurement only) ---
    model['pll'] = {
        'PLL1': [
            ['name', 'T_filter', 'bus'],
            ['PLL_B8', 0.1, 'B8'],
        ],
    }

    # --- Build system ---
    ps = dps.PowerSystemModel(model=model)

    # --- Power flow + initialization ---
    ps.power_flow()
    ps.init_dyn_sim()

    # --- Simulation settings ---
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

    # --- Bus index for B8 ---
    bus_names = list(ps.buses['name'])
    k = bus_names.index('B8')

    # --- Storage ---
    result_dict = defaultdict(list)
    P_e_stored = []
    E_f_stored = []
    V_stored = []
    PLL_ang = []
    PLL_freq = []

    t = 0.0
    t0 = time.time()

    # =========================================================
    # TIME SIMULATION LOOP (NO FAULT)
    # =========================================================
    while t < t_end:

        # No fault
        ps.y_bus_red_mod[k, k] = 0.0

        sol.step()

        x = sol.y
        v = sol.v
        t = sol.t

        # Store time
        result_dict[('Global', 't')].append(t)

        # Store states
        for desc, state in zip(ps.state_desc, x):
            result_dict[tuple(desc)].append(state)

        # Generator outputs
        P_e_stored.append(ps.gen['GEN'].P_e(x, v).copy())
        E_f_stored.append(ps.gen['GEN'].E_f(x, v).copy())
        V_stored.append(float(np.abs(v[k])))

        # PLL outputs
        PLL_ang.append(ps.pll['PLL1'].output(x, v))
        PLL_freq.append(ps.pll['PLL1'].freq_est(x, v))

    print(f"Simulation completed in {time.time() - t0:.2f} seconds")

    # =========================================================
    # POST PROCESSING
    # =========================================================
    result = pd.DataFrame({kk: pd.Series(vv) for kk, vv in result_dict.items()})
    result.columns = pd.MultiIndex.from_tuples(result.columns)

    time_series = result[('Global', 't')].to_numpy()

    speed_df = result.xs(key='speed', axis='columns', level=1)
    angle_df = result.xs(key='angle', axis='columns', level=1)

    P_e = np.array(P_e_stored)
    E_f = np.array(E_f_stored)
    V_mag = np.array(V_stored)
    PLL_ang = np.unwrap(np.array(PLL_ang).flatten())
    PLL_freq = np.array(PLL_freq).flatten()

    gen_rows = model['generators']['GEN'][1:]
    gen_names = [row[0] for row in gen_rows]
    gen_Sn = np.array([row[2] for row in gen_rows], dtype=float)

    # =========================================================
    # PLOTS
    # =========================================================

    fig, ax = plt.subplots(3, figsize=(9, 7))
    fig.suptitle("Generator Speed, Angle and Electrical Power")

    ax[0].plot(time_series, speed_df.to_numpy())
    ax[0].set_ylabel("Speed (p.u.)")
    ax[0].legend(gen_names)

    ax[1].plot(time_series, angle_df.to_numpy())
    ax[1].set_ylabel("Angle (rad)")
    ax[1].legend(gen_names)

    P_e_pu = P_e / gen_Sn
    ax[2].plot(time_series, P_e_pu)
    ax[2].set_ylabel("P_e (p.u.)")
    ax[2].set_xlabel("Time (s)")
    ax[2].legend([f"{g} P_e" for g in gen_names])

    plt.figure()
    plt.plot(time_series, V_mag)
    plt.title("Voltage magnitude at B8")
    plt.xlabel("Time (s)")
    plt.ylabel("|V| (p.u.)")

    plt.figure()
    plt.plot(time_series, PLL_ang)
    plt.title("PLL Angle Estimate (B8)")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")

    plt.figure()
    plt.plot(time_series, PLL_freq)
    plt.title("PLL Frequency Estimate (B8)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency deviation")

    plt.show()

    print("Minimum voltage at B8 =", np.min(V_mag))