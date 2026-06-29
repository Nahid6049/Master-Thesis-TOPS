# simulate_pll_compare.py
# ------------------------------------------------------------
# Strong vs Weak Grid PLL Comparison
# Same network, same fault, same PLL location (B8)
# Only difference: Thevenin impedance for weak case
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import time

import tops.dynamic as dps
import tops.solvers as dps_sol
import my_network


# ============================================================
# Weak grid modifier (Thevenin impedance behind slack)
# ============================================================

def make_weak_grid(model):

    old_slack = model.get('slack_bus', 'B1')
    new_slack = old_slack + "_W"

    # --- Add new bus ---
    buses = model['buses']
    header = buses[0]
    idx_name = header.index('name')
    idx_vn = header.index('V_n')

    Vn_old = None
    for row in buses[1:]:
        if row[idx_name] == old_slack:
            Vn_old = row[idx_vn]
            break

    buses.append([new_slack, Vn_old])
    model['buses'] = buses

    model['slack_bus'] = new_slack

    # --- Thevenin impedance (main weakness knob) ---
    X_th = 0.30
    R_th = 0.0

    lines = model['lines']
    lines.append([
        'L_TH',
        new_slack,
        old_slack,
        1,
        1000,
        0,
        'p.u.',
        R_th,
        X_th,
        0.0
    ])

    model['lines'] = lines

    return model


# ============================================================
# Simulation function (used for both cases)
# ============================================================

def run_simulation(model, label):

    # Add PLL at B8
    model['pll'] = {
        'PLL1': [
            ['name', 'T_filter', 'bus'],
            ['PLL_B8', 0.05, 'B8'],
        ],
    }

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

    bus_names = list(ps.buses['name'])
    k = bus_names.index('B8')

    t_store = []
    V_mag = []
    PLL_freq = []
    PLL_ang = []

    t = 0.0

    while t < t_end:

        # Same fault for both cases
        if 1.0 < t < 1.05:
            ps.y_bus_red_mod[k, k] = 10000
        else:
            ps.y_bus_red_mod[k, k] = 0

        sol.step()

        x = sol.y
        v = sol.v
        t = sol.t

        t_store.append(t)
        V_mag.append(abs(v[k]))

        freq = ps.pll['PLL1'].freq_est(x, v)
        ang = ps.pll['PLL1'].output(x, v)

        PLL_freq.append(float(np.array(freq).flatten()[0]))
        PLL_ang.append(float(np.array(ang).flatten()[0]))

    return np.array(t_store), np.array(V_mag), np.array(PLL_freq), np.array(PLL_ang)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ---------- STRONG ----------
    model_strong = my_network.load()
    t_s, V_s, f_s, ang_s = run_simulation(model_strong, "Strong")

    # ---------- WEAK ----------
    model_weak = my_network.load()
    model_weak = make_weak_grid(model_weak)
    t_w, V_w, f_w, ang_w = run_simulation(model_weak, "Weak")

    # ============================================================
    # PLOTS
    # ============================================================

    # Voltage comparison
    plt.figure()
    plt.plot(t_s, V_s, label="Strong")
    plt.plot(t_w, V_w, '--', label="Weak")
    plt.title("Voltage at B8")
    plt.xlabel("Time (s)")
    plt.ylabel("|V| (p.u.)")
    plt.legend()
    plt.grid()

    # Frequency comparison (zoom)
    mask_s = (t_s > 0.8) & (t_s < 2.0)
    mask_w = (t_w > 0.8) & (t_w < 2.0)

    plt.figure()
    plt.plot(t_s[mask_s], f_s[mask_s], label="Strong")
    plt.plot(t_w[mask_w], f_w[mask_w], '--', label="Weak")
    plt.title("PLL Frequency Comparison (Zoom 0.8–2.0 s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency deviation")
    plt.legend()
    plt.grid()

    # Angle comparison (zoom)
    ang_s_rel = ang_s - ang_s[0]
    ang_w_rel = ang_w - ang_w[0]

    mask_s = (t_s > 0.8) & (t_s < 1.5)
    mask_w = (t_w > 0.8) & (t_w < 1.5)

    plt.figure()
    plt.plot(t_s[mask_s], ang_s_rel[mask_s], label="Strong")
    plt.plot(t_w[mask_w], ang_w_rel[mask_w], '--', label="Weak")
    plt.title("PLL Angle Comparison (Zoom 0.8–1.5 s)")
    plt.xlabel("Time (s)")
    plt.ylabel("ΔAngle (rad)")
    plt.legend()
    plt.grid()

    plt.show()
print("Minimum voltage at B8 (Strong) =", np.min(V_s))
print("Minimum voltage at B8 (Weak)   =", np.min(V_w))