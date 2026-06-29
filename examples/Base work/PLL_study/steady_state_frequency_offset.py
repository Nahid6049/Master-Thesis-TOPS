import tops
print(tops.__file__)
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


# =====================
# SETTINGS
# =====================
PLL_BUS = "B8"
T_END = 50.0
MAX_STEP = 5e-3
F_NOM = 50.0

# Permanent imbalance (applied from start, via extra shunt admittance for Z-loads)
LOAD_STEP_FACTOR = 0.10       # +10% (increase to 0.2 if you want larger deviation)
LOAD_BUSES = ["B5", "B6"]

# Case 3 damping value (to allow a true steady offset when governor is OFF)
D_STEADY = 2.0


# ----------------------------
# Helpers
# ----------------------------
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
    else:
        raise ValueError("pll_type must be 'PLL1' or 'PLL2'")

    return model


def modify_generator_damping(model, new_D):
    """
    Set D for all generators in model['generators']['GEN'] table.
    """
    model = dict(model)
    gen_table = model["generators"]["GEN"]
    header = gen_table[0]
    iD = header.index("D")

    new_table = [header]
    for r in gen_table[1:]:
        rr = list(r)
        rr[iD] = float(new_D)
        new_table.append(rr)

    model["generators"]["GEN"] = new_table
    return model


def get_reduced_idx(ps, full_idx):
    for attr in ("k_bus_red", "bus_red_map", "bus_reduction_idx", "k_red"):
        if hasattr(ps, attr):
            m = getattr(ps, attr)
            try:
                return int(m[full_idx])
            except Exception:
                pass
    return full_idx


def get_Zload_S_pu_by_bus(model):
    """
    Returns S (pu) for each bus that has Z-load in model['loads'].
    """
    base_mva = float(model.get("base_mva", 1000.0))
    loads = model.get("loads", [])
    if not loads or len(loads) < 2:
        return {}

    header = loads[0]
    i_bus = header.index("bus")
    i_P = header.index("P")
    i_Q = header.index("Q")
    i_model = header.index("model")

    Sbus = {}
    for row in loads[1:]:
        bus = row[i_bus]
        P = float(row[i_P]) / base_mva
        Q = float(row[i_Q]) / base_mva
        mdl = str(row[i_model]).strip()
        if mdl.upper() == "Z":
            Sbus[bus] = Sbus.get(bus, 0.0 + 0.0j) + (P + 1j * Q)
    return Sbus


def get_speed_indices(ps):
    idx = []
    for i, desc in enumerate(ps.state_desc):
        if len(desc) >= 2 and str(desc[1]).lower() == "speed":
            idx.append(i)
    if not idx:
        raise RuntimeError("No speed states found -> cannot compute system frequency.")
    return np.array(idx, dtype=int)


def apply_permanent_Zload_increase(ps, v_now, Sbus, load_bus_full, load_bus_red):
    """
    Adds extra shunt admittance at the load buses to represent +X% Z-load permanently.
    We keep y_diag_orig + y_added enforced at each time step.
    """
    y_diag_orig = {}
    y_added = {}

    # store original diagonals
    for b, kr in zip(LOAD_BUSES, load_bus_red):
        y_diag_orig[b] = ps.y_bus_red_mod[kr, kr].copy()
        y_added[b] = 0.0 + 0.0j

    # compute extra admittance based on initial V magnitude
    for b, k_full, k_red in zip(LOAD_BUSES, load_bus_full, load_bus_red):
        if b not in Sbus:
            continue

        Vmag = float(np.abs(v_now[k_full]))
        if Vmag < 1e-6:
            Vmag = 1e-6

        S = Sbus[b]  # pu
        # For Z-load: S = V^2 * conj(Y) => Y = conj(S)/V^2
        Y_base = np.conj(S) / (Vmag ** 2)
        dY = LOAD_STEP_FACTOR * Y_base

        y_added[b] = dY
        ps.y_bus_red_mod[k_red, k_red] = y_diag_orig[b] + dY

    return y_diag_orig, y_added


def run_case(pll_type, governor=True, damping=None):
    """
    Runs one simulation with:
      - permanent +Z load increase from the start
      - optional governor removal
      - optional generator damping override
    Returns arrays: t, f_sys, f_pll, ang_err
    """
    model = my_network.load()
    model = attach_pll(model, pll_type)

    # Optional: avoid VSC warning if your user_lib lacks VSC_PQ
    model.pop("vsc", None)

    if not governor:
        model.pop("gov", None)

    if damping is not None:
        model = modify_generator_damping(model, damping)

    # For computing dY from Z-loads
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
    k_pll_full = bus_names.index(PLL_BUS)

    # map load buses
    load_bus_full = []
    load_bus_red = []
    for b in LOAD_BUSES:
        if b not in bus_names:
            raise ValueError(f"Load bus {b} not found. Available: {bus_names}")
        kf = bus_names.index(b)
        kr = get_reduced_idx(ps, kf)
        load_bus_full.append(kf)
        load_bus_red.append(kr)

    speed_idx = get_speed_indices(ps)

    # do one step so sol.v exists (algebraic solved)
    sol.step()

    # apply permanent load increase based on initial voltages
    y_diag_orig, y_added = apply_permanent_Zload_increase(
        ps, sol.v, Sbus, load_bus_full, load_bus_red
    )

    t_store = []
    f_sys_store = []
    f_pll_store = []
    ang_err_store = []

    def store_sample():
        x = sol.y
        v = sol.v
        t_store.append(sol.t)

        # system frequency from mean speed deviation (Δω)
        d_omega = float(np.mean(x[speed_idx]))
        f_sys = F_NOM * (1.0 + d_omega)

        # PLL frequency estimate (pu deviation)
        f_est = ps.pll[pll_type].freq_est(x, v)
        f_est = float(np.array(f_est).flatten()[0])
        f_pll = F_NOM * (1.0 + f_est)

        # angle error
        theta_v = float(np.angle(v[k_pll_full]))
        theta_pll = ps.pll[pll_type].output(x, v)
        theta_pll = float(np.array(theta_pll).flatten()[0])
        ang_err = wrap_to_pi(theta_pll - theta_v)

        f_sys_store.append(f_sys)
        f_pll_store.append(f_pll)
        ang_err_store.append(ang_err)

    # store first sample after the first step + load change
    store_sample()

    while sol.t < T_END:
        # enforce permanent load change
        for b, k_red in zip(LOAD_BUSES, load_bus_red):
            ps.y_bus_red_mod[k_red, k_red] = y_diag_orig[b] + y_added[b]

        sol.step()
        store_sample()

    return np.array(t_store), np.array(f_sys_store), np.array(f_pll_store), np.array(ang_err_store)


def summarize_steady(t, f_sys, f_pll, ang_err, label, window_s=5.0):
    ss = t >= (t[-1] - window_s)
    f_err = f_pll[ss] - f_sys[ss]
    return {
        "label": label,
        "f_sys_ss": float(np.mean(f_sys[ss])),
        "f_pll_ss": float(np.mean(f_pll[ss])),
        "f_err_rms": float(np.sqrt(np.mean(f_err**2))),
        "ang_err_mean": float(np.mean(ang_err[ss])),
        "ang_err_rms": float(np.sqrt(np.mean(ang_err[ss]**2))),
    }


# =============================
# RUN 3 CASES
# =============================
cases = [
    ("Case 1: Governor ON", True,  None),
    ("Case 2: Governor OFF (D=0)", False, 0.0),
    ("Case 3: Governor OFF (D>0 steady)", False, D_STEADY),
]

all_results = {}

for case_name, gov_on, damp in cases:
    print(f"\n--- {case_name} ---")

    t1, f_sys1, f_pll1, ang1 = run_case("PLL1", governor=gov_on, damping=damp)
    t2, f_sys2, f_pll2, ang2 = run_case("PLL2", governor=gov_on, damping=damp)

    all_results[case_name] = (t1, f_sys1, f_pll1, ang1, f_pll2, ang2)

    s1 = summarize_steady(t1, f_sys1, f_pll1, ang1, "PLL1")
    s2 = summarize_steady(t1, f_sys1, f_pll2, ang2, "PLL2")

    print("Steady window (last 5s) summary:")
    print(f"  PLL1: f_sys={s1['f_sys_ss']:.4f} Hz, f_pll={s1['f_pll_ss']:.4f} Hz, "
          f"RMS freq err={s1['f_err_rms']:.3e} Hz, mean phase offset={s1['ang_err_mean']:.3e} rad")
    print(f"  PLL2: f_sys={s2['f_sys_ss']:.4f} Hz, f_pll={s2['f_pll_ss']:.4f} Hz, "
          f"RMS freq err={s2['f_err_rms']:.3e} Hz, mean phase offset={s2['ang_err_mean']:.3e} rad")


# =============================
# PLOTS
# =============================
for case_name, data in all_results.items():
    t, f_sys, f_pll1, ang1, f_pll2, ang2 = data

    plt.figure()
    plt.plot(t, f_sys, label="System frequency")
    plt.plot(t, f_pll1, label="PLL1 estimate")
    plt.plot(t, f_pll2, "--", label="PLL2 estimate")
    plt.title(case_name + " — Frequency")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(t, ang1, label="PLL1 phase error (rad)")
    plt.plot(t, ang2, "--", label="PLL2 phase error (rad)")
    plt.title(case_name + " — Phase tracking error")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle error (rad)")
    plt.grid(True)
    plt.legend()

plt.show()