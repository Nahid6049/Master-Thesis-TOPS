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
T_END = 20.0
T_EVENT = 10.0
MAX_STEP = 5e-3
GEN_TO_TRIP = "G3"
F_NOM = 50.0


# ----------------------------
# Helper
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


def set_generator_P(model, gen_name, new_P):
    model = dict(model)
    gens = model["generators"]["GEN"]
    header = gens[0]
    rows = [list(r) for r in gens[1:]]

    i_name = header.index("name")
    i_P = header.index("P")

    for r in rows:
        if r[i_name] == gen_name:
            r[i_P] = float(new_P)

    model["generators"]["GEN"] = [header] + rows
    return model


def get_speed_indices(ps):
    idx = []
    for i, desc in enumerate(ps.state_desc):
        if len(desc) >= 2 and str(desc[1]).lower() == "speed":
            idx.append(i)
    return np.array(idx)


def run_segment(model, pll_type, t0, t1, x0=None):

    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_lib)
    ps.power_flow()
    ps.init_dyn_sim()

    if x0 is not None:
        ps.x0 = x0.copy()

    sol = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        t0,
        ps.x0.copy(),
        t1,
        max_step=MAX_STEP,
    )

    speed_idx = get_speed_indices(ps)
    bus_names = list(ps.buses["name"])
    k_pll = bus_names.index(PLL_BUS)

    t_store = []
    f_sys_store = []
    f_pll_store = []
    ang_err_store = []

    while sol.t < t1:

        sol.step()

        x = sol.y
        v = sol.v
        t_store.append(sol.t)

        # ----- System frequency from Δω -----
        d_omega = np.mean(x[speed_idx])
        f_sys = F_NOM * (1.0 + d_omega)
        f_sys_store.append(float(f_sys))

        # ----- PLL frequency -----
        f_est = ps.pll[pll_type].freq_est(x, v)
        f_est = float(np.array(f_est).flatten()[0])
        f_pll = F_NOM * (1.0 + f_est)
        f_pll_store.append(float(f_pll))

        # ----- Angle error -----
        theta_v = float(np.angle(v[k_pll]))
        theta_pll = ps.pll[pll_type].output(x, v)
        theta_pll = float(np.array(theta_pll).flatten()[0])

        ang_err = wrap_to_pi(theta_pll - theta_v)
        ang_err_store.append(ang_err)

    return (
        np.array(t_store),
        np.array(f_sys_store),
        np.array(f_pll_store),
        np.array(ang_err_store),
        sol.y.copy()
    )


def run_case(pll_type):

    # -------- Stage 1: Normal --------
    model1 = my_network.load()
    model1 = attach_pll(model1, pll_type)
    model1.pop("vsc", None)

    t1, f_sys1, f_pll1, ang_err1, x_end = run_segment(
        model1, pll_type, 0.0, T_EVENT
    )

    # -------- Stage 2: Trip --------
    model2 = my_network.load()
    model2 = attach_pll(model2, pll_type)
    model2.pop("vsc", None)
    model2 = set_generator_P(model2, GEN_TO_TRIP, 0.0)

    t2, f_sys2, f_pll2, ang_err2, _ = run_segment(
        model2, pll_type, T_EVENT, T_END, x_end
    )

    # Stitch
    t = np.concatenate([t1, t2])
    f_sys = np.concatenate([f_sys1, f_sys2])
    f_pll = np.concatenate([f_pll1, f_pll2])
    ang_err = np.concatenate([ang_err1, ang_err2])

    idx = t >= T_EVENT

    max_freq_err = np.max(np.abs(f_pll[idx] - f_sys[idx]))
    rms_freq_err = np.sqrt(np.mean((f_pll[idx] - f_sys[idx])**2))

    max_ang_err = np.max(np.abs(ang_err[idx]))
    rms_ang_err = np.sqrt(np.mean(ang_err[idx]**2))

    return t, f_sys, f_pll, ang_err, max_freq_err, rms_freq_err, max_ang_err, rms_ang_err


# =====================
# MAIN
# =====================
if __name__ == "__main__":

    print("Running PLL1...")
    t1, f_sys1, f_pll1, ang_err1, maxf1, rmsf1, maxa1, rmsa1 = run_case("PLL1")

    print("Running PLL2...")
    t2, f_sys2, f_pll2, ang_err2, maxf2, rmsf2, maxa2, rmsa2 = run_case("PLL2")

    # ----- Plot Frequency -----
    plt.figure()
    plt.plot(t1, f_sys1, label="System frequency")
    plt.plot(t1, f_pll1, label="PLL1")
    plt.plot(t2, f_pll2, "--", label="PLL2")
    plt.axvline(T_EVENT, color="r", linestyle="--")
    plt.title("Frequency Tracking After Generator Power Loss")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.grid(True)
    plt.legend()

    # ----- Plot Angle Error -----
    plt.figure()
    plt.plot(t1, ang_err1, label="PLL1 angle error (rad)")
    plt.plot(t2, ang_err2, "--", label="PLL2 angle error (rad)")
    plt.axvline(T_EVENT, color="r", linestyle="--")
    plt.title("PLL Angle Tracking Error")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle error (rad)")
    plt.grid(True)
    plt.legend()

    plt.show()

    # ----- Print Metrics -----
    print("\n================ Frequency Error =================")
    print(f"{'Metric':<28}{'PLL1':<15}{'PLL2':<15}")
    print("-----------------------------------------------")
    print(f"{'Max |f_PLL - f_sys| (Hz)':<28}{maxf1:<15.6e}{maxf2:<15.6e}")
    print(f"{'RMS |f_PLL - f_sys| (Hz)':<28}{rmsf1:<15.6e}{rmsf2:<15.6e}")

    print("\n================ Angle Error =================")
    print(f"{'Metric':<28}{'PLL1':<15}{'PLL2':<15}")
    print("-----------------------------------------------")
    print(f"{'Max |θ_PLL-θ_V| (rad)':<28}{maxa1:<15.6e}{maxa2:<15.6e}")
    print(f"{'RMS |θ_PLL-θ_V| (rad)':<28}{rmsa1:<15.6e}{rmsa2:<15.6e}")
    # ============================
# NUMERICAL COMPARISON PLL1 vs PLL2
# ============================
import numpy as np

def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))

def settling_time(t, err, t_event, band, hold_s=1.0):
    """
    First time after t_event when |err| stays <= band for at least hold_s.
    Returns np.nan if never settles.
    """
    idx = np.where(t >= t_event)[0]
    if idx.size == 0:
        return float("nan")

    dt = np.median(np.diff(t))
    hold_n = max(1, int(round(hold_s / dt)))

    e = np.abs(err)
    for k in idx:
        if k + hold_n >= len(t):
            break
        if np.all(e[k:k+hold_n] <= band):
            return float(t[k] - t_event)
    return float("nan")

# Masks after event
m1 = t1 >= T_EVENT
m2 = t2 >= T_EVENT

# Frequency errors
ferr1 = f_pll1 - f_sys1
ferr2 = f_pll2 - f_sys2

max_ferr1 = float(np.max(np.abs(ferr1[m1])))
rms_ferr1 = rms(ferr1[m1])

max_ferr2 = float(np.max(np.abs(ferr2[m2])))
rms_ferr2 = rms(ferr2[m2])

# Angle errors (already computed as wrapped)
max_aerr1 = float(np.max(np.abs(ang_err1[m1])))
rms_aerr1 = rms(ang_err1[m1])

max_aerr2 = float(np.max(np.abs(ang_err2[m2])))
rms_aerr2 = rms(ang_err2[m2])

# Settling times (choose bands you like)
# Example: freq error settles within 0.01 Hz, angle error within 0.01 rad
ts_f1 = settling_time(t1, ferr1, T_EVENT, band=0.01, hold_s=1.0)
ts_f2 = settling_time(t2, ferr2, T_EVENT, band=0.01, hold_s=1.0)

ts_a1 = settling_time(t1, ang_err1, T_EVENT, band=0.01, hold_s=1.0)
ts_a2 = settling_time(t2, ang_err2, T_EVENT, band=0.01, hold_s=1.0)

# ROCOF (system frequency derivative) after event
dfdt1 = np.gradient(f_sys1, t1)
dfdt2 = np.gradient(f_sys2, t2)
max_rocof1 = float(np.max(np.abs(dfdt1[m1])))
max_rocof2 = float(np.max(np.abs(dfdt2[m2])))

print("\n================ Frequency Tracking Performance =================")
print(f"{'Metric':<40}{'PLL1':<15}{'PLL2':<15}")
print("-"*70)
print(f"{'Peak Frequency Tracking Error (Hz)':<40}{maxf1:<15.6e}{maxf2:<15.6e}")
print(f"{'Average Frequency Tracking Error (RMS, Hz)':<40}{rmsf1:<15.6e}{rmsf2:<15.6e}")

print("\n================ Phase Tracking Performance =================")
print(f"{'Metric':<40}{'PLL1':<15}{'PLL2':<15}")
print("-"*70)
print(f"{'Peak Phase Tracking Error (rad)':<40}{maxa1:<15.6e}{maxa2:<15.6e}")
print(f"{'Average Phase Tracking Error (RMS, rad)':<40}{rmsa1:<15.6e}{rmsa2:<15.6e}")