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
# Make user_lib (package folder) visible
#   Your user_lib folder is: D:\...\TOPS-main\TOPS-main\user_lib
#   => Add its parent directory to sys.path
# ----------------------------
sys.path.append(r"D:\Masters REM+\Master Thesis\thesis work\TOPS-main\TOPS-main")

import tops.dynamic as dps
import tops.solvers as dps_sol
import my_network
import user_lib   # <-- required so PLL block is recognized

# ============================
# SETTINGS
# ============================
PLL_BUS = "B8"
FAULT_BUS = "B8"
T_FAULT_START = 1.0
T_FAULT_END = 1.05
T_END = 10.0
FAULT_ADMITTANCE = 10000.0    # p.u., large to emulate three-phase bolted fault
T_FILTER = 0.05               # PLL filter time constant
MAX_STEP = 5e-3
ERR_THR = 0.05                # rad, angle error settling threshold

# Weak grid impedance (Thevenin)
X_TH_WEAK = 0.30   # try 0.6 for very weak
R_TH_WEAK = 0.0

# ============================
# WEAK GRID MODIFIER
# ============================
def make_weak_grid(model, X_th=0.3, R_th=0.0):
    old_slack = model.get("slack_bus", "B1")
    new_slack = old_slack + "_W"

    # find Vn of old slack
    buses = model["buses"]
    header = buses[0]
    idx_name = header.index("name")
    idx_vn = header.index("V_n")

    Vn_old = None
    for row in buses[1:]:
        if row[idx_name] == old_slack:
            Vn_old = row[idx_vn]
            break

    if Vn_old is None:
        raise ValueError(f"Slack bus {old_slack} not found in 'buses'.")

    # add new slack bus and set it as slack
    buses.append([new_slack, Vn_old])
    model["buses"] = buses
    model["slack_bus"] = new_slack

    # add Thevenin line from new slack to original slack in p.u. units
    lines = model["lines"]
    # Expecting the 'lines' to already have a header at index 0
    lines.append([
        "L_TH",        # name
        new_slack,     # from_bus
        old_slack,     # to_bus
        1,             # length (not used in p.u. mode)
        1000,          # S_n (not used in p.u. mode)
        0,             # V_n (0 indicates p.u. data in your dataset)
        "p.u.",        # unit
        R_th,          # R (p.u.)
        X_th,          # X (p.u.)
        0.0            # B (p.u.)
    ])
    model["lines"] = lines

    return model

# ============================
# Helper: map full bus index -> reduced Y-bus index (if reduction mapping exists)
# ============================
def _get_reduced_idx(ps, full_idx):
    """
    Returns the row/col index in ps.y_bus_red_mod corresponding to a full bus index.
    If no explicit mapping is available, assumes same indexing.
    """
    # Some TOPS builds have ps.k_bus_red or ps.bus_red_map. Try common names safely.
    for attr in ("k_bus_red", "bus_red_map", "bus_reduction_idx", "k_red"):
        if hasattr(ps, attr):
            m = getattr(ps, attr)
            # array or dict-like:
            try:
                return int(m[full_idx])
            except Exception:
                pass
    # Fallback: assume same index (works if y_bus_red_mod keeps full bus layout)
    return full_idx

# ============================
# RUN CASE
# ============================
def run_case(weak=False):

    model = my_network.load()

    if weak:
        model = make_weak_grid(model, X_th=X_TH_WEAK, R_th=R_TH_WEAK)

    # Attach PLL block (group key "PLL1", one instance named by bus)
    model["pll"] = {
        "PLL1": [
            ["name", "T_filter", "bus"],
            [f"PLL_{PLL_BUS}", float(T_FILTER), PLL_BUS],
        ]
    }

    # IMPORTANT: pass user_mdl_lib so PLL model is available
    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_lib)

    # Power flow + dynamic init
    ps.power_flow()
    ps.init_dyn_sim()

    # Create solver
    sol = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0.0,
        ps.x0.copy(),
        T_END,
        max_step=MAX_STEP,
    )

    # --- indices ---
    bus_names = list(ps.buses["name"])
    if PLL_BUS not in bus_names:
        raise ValueError(f"PLL bus {PLL_BUS} not found. Available: {bus_names}")
    if FAULT_BUS not in bus_names:
        raise ValueError(f"Fault bus {FAULT_BUS} not found. Available: {bus_names}")

    k_pll_full = bus_names.index(PLL_BUS)
    k_fault_full = bus_names.index(FAULT_BUS)

    # y-bus reduced index (map if needed)
    k_fault_red = _get_reduced_idx(ps, k_fault_full)

    # Store original diagonal to restore after fault
    y_diag_orig = ps.y_bus_red_mod[k_fault_red, k_fault_red].copy()

    # --- storage ---
    t_store, V_mag, V_ang, PLL_ang, PLL_freq = [], [], [], [], []

    # --- simulate ---
    while sol.t < T_END:

        # Apply/clear fault by modifying the reduced Y-bus diagonal at the proper index
        if T_FAULT_START < sol.t < T_FAULT_END:
            ps.y_bus_red_mod[k_fault_red, k_fault_red] = y_diag_orig + FAULT_ADMITTANCE
        else:
            ps.y_bus_red_mod[k_fault_red, k_fault_red] = y_diag_orig

        sol.step()

        x = sol.y
        v = sol.v
        t = sol.t

        t_store.append(t)

        Vpll = v[k_pll_full]
        V_mag.append(float(np.abs(Vpll)))
        V_ang.append(float(np.angle(Vpll)))

        # Group "PLL1" may have multiple instances; use outputs for the instance vector
        f_est = ps.pll["PLL1"].freq_est(x, v)
        th_est = ps.pll["PLL1"].output(x, v)

        PLL_freq.append(float(np.array(f_est).flatten()[0]))
        PLL_ang.append(float(np.array(th_est).flatten()[0]))

    # --- post ---
    t_store = np.array(t_store)
    V_mag = np.array(V_mag)
    V_ang = np.unwrap(np.array(V_ang))
    PLL_ang = np.unwrap(np.array(PLL_ang))
    PLL_freq = np.array(PLL_freq)

    err = PLL_ang - V_ang

    min_voltage = float(np.min(V_mag))
    max_err = float(np.max(np.abs(err)))
    rms_err = float(np.sqrt(np.mean(err**2)))
    max_freq = float(np.max(np.abs(PLL_freq)))

    # settling time (first time after fault clear where |err| stays < ERR_THR)
    settling = np.nan
    idx_post = np.where(t_store >= T_FAULT_END)[0]
    if len(idx_post) > 0:
        epost = np.abs(err[idx_post])
        for i in range(len(epost)):
            if np.all(epost[i:] < ERR_THR):
                settling = float(t_store[idx_post[i]] - T_FAULT_END)
                break

    return {
        "t": t_store,
        "err": err,
        "PLLfreq": PLL_freq,
        "minV": min_voltage,
        "maxErr": max_err,
        "rmsErr": rms_err,
        "maxFreq": max_freq,
        "settle": settling,
    }

# ============================
# MAIN
# ============================
if __name__ == "__main__":

    print("\n======================================")
    print("PLL Weak Grid Sensitivity Study")
    print("======================================")

    strong = run_case(weak=False)
    weak = run_case(weak=True)

    print("\n--- STRONG GRID ---")
    print(f"Min |V|: {strong['minV']:.4f}")
    print(f"Max angle error: {strong['maxErr']:.4f} rad")
    print(f"RMS error: {strong['rmsErr']:.4f} rad")
    print(f"Max freq deviation: {strong['maxFreq']:.4f} (rad/s or p.u.)")
    print(f"Settling time: {strong['settle']:.4f} s")

    print("\n--- WEAK GRID ---")
    print(f"Min |V|: {weak['minV']:.4f}")
    print(f"Max angle error: {weak['maxErr']:.4f} rad")
    print(f"RMS error: {weak['rmsErr']:.4f} rad")
    print(f"Max freq deviation: {weak['maxFreq']:.4f} (rad/s or p.u.)")
    print(f"Settling time: {weak['settle']:.4f} s")

    # Angle error comparison
    plt.figure()
    plt.plot(strong["t"], strong["err"], label="Strong Grid")
    plt.plot(weak["t"], weak["err"], "--", label="Weak Grid")
    plt.axvspan(T_FAULT_START, T_FAULT_END, color="red", alpha=0.25)
    plt.title("PLL Angle Error: Strong vs Weak Grid")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle error (rad)")
    
    plt.legend()

    # Frequency comparison (zoom)
    plt.figure()
    mask = (strong["t"] > 0.8) & (strong["t"] < 2.0)
    plt.plot(strong["t"][mask], strong["PLLfreq"][mask], label="Strong Grid")
    plt.plot(weak["t"][mask],   weak["PLLfreq"][mask],   "--", label="Weak Grid")
    plt.axvspan(T_FAULT_START, T_FAULT_END, color="red", alpha=0.25)
    plt.title("PLL Frequency Response: Strong vs Weak Grid")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency deviation")
    
    plt.legend()

    plt.show()