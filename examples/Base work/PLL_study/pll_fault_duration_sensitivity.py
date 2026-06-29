# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import importlib

# ============================
# PATHS
# ============================
sys.path.append(r"D:\Masters REM+\Master Thesis\paper\TOPS-main\TOPS-main")
sys.path.append(r"D:\Masters REM+\Master Thesis\paper\TOPS-main\TOPS-main\examples\Base work")
sys.path.append(r"D:\Masters REM+\Master Thesis\paper\TOPS-main\TOPS-main\examples\user_models")

import tops.dynamic as dps
import tops.solvers as dps_sol
import user_lib
import my_network


# ============================
# SETTINGS
# ============================
PLL_BUS = "B8"
FAULT_BUS = "B8"

T_FAULT_START = 1.0
T_END = 10.0
FAULT_ADMITTANCE = 10000.0

MAX_STEP = 5e-3
ERR_THR = 0.05

DURATIONS = [0.05, 0.10, 0.20]
T_FILTER = 0.05


# ============================
# RUN ONE CASE
# ============================
def run_case(duration):

    importlib.reload(my_network)
    model = my_network.load()

    t_fault_end = T_FAULT_START + duration

    # Add external PLL measurement block
    model["pll"] = {
        "PLL1": [
            ["name", "T_filter", "bus"],
            [f"PLL_{PLL_BUS}", float(T_FILTER), PLL_BUS],
        ]
    }

    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_lib)
    ps.power_flow()
    ps.init_dyn_sim()

    try:
        x0 = ps.x_0.copy()
    except:
        x0 = ps.x0.copy()

    sol = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0.0,
        x0,
        T_END,
        max_step=MAX_STEP,
    )

    bus_names = list(ps.buses["name"])
    k_pll = bus_names.index(PLL_BUS)
    k_fault = bus_names.index(FAULT_BUS)

    t_store = []
    V_mag = []
    V_ang = []
    PLL_ang = []
    PLL_freq = []

    P_vsc = []
    Q_vsc = []
    I_vsc = []
    VSC_freq = []
    VSC_rocof = []

    t = 0.0

    while t < T_END:

        if T_FAULT_START < t < t_fault_end:
            ps.y_bus_red_mod[k_fault, k_fault] = FAULT_ADMITTANCE
        else:
            ps.y_bus_red_mod[k_fault, k_fault] = 0.0

        sol.step()

        t = sol.t
        x = sol.y
        v = sol.v

        Vpll = v[k_pll]

        t_store.append(t)
        V_mag.append(float(np.abs(Vpll)))
        V_ang.append(float(np.angle(Vpll)))

        # External PLL response
        f_est = ps.pll["PLL1"].freq_est(x, v)
        th_est = ps.pll["PLL1"].output(x, v)

        PLL_freq.append(float(np.array(f_est).flatten()[0]))
        PLL_ang.append(float(np.array(th_est).flatten()[0]))

        # VSC response
        vsc = ps.vsc["VSC_SI"]

        p_vsc = vsc.p_e(x, v)
        q_vsc = vsc.q_e(x, v)
        i_vsc = vsc.i_inj(x, v)
        f_vsc = vsc.freq_est(x, v)
        rocof_vsc = vsc.rocof_est(x, v)

        P_vsc.append(float(np.array(p_vsc).flatten()[0]))
        Q_vsc.append(float(np.array(q_vsc).flatten()[0]))
        I_vsc.append(float(np.abs(np.array(i_vsc).flatten()[0])))
        VSC_freq.append(float(np.array(f_vsc).flatten()[0]))
        VSC_rocof.append(float(np.array(rocof_vsc).flatten()[0]))

    t_store = np.array(t_store)
    V_mag = np.array(V_mag)
    V_ang = np.unwrap(np.array(V_ang))
    PLL_ang = np.unwrap(np.array(PLL_ang))
    PLL_freq = np.array(PLL_freq)

    P_vsc = np.array(P_vsc)
    Q_vsc = np.array(Q_vsc)
    I_vsc = np.array(I_vsc)
    VSC_freq = np.array(VSC_freq)
    VSC_rocof = np.array(VSC_rocof)

    err = PLL_ang - V_ang

    settling = np.nan
    idx_post = np.where(t_store >= t_fault_end)[0]

    if len(idx_post) > 0:
        epost = np.abs(err[idx_post])
        for i in range(len(epost)):
            if np.all(epost[i:] < ERR_THR):
                settling = float(t_store[idx_post[i]] - t_fault_end)
                break

    return {
        "duration": duration,
        "t_fault_end": t_fault_end,
        "t": t_store,
        "Vmag": V_mag,
        "err": err,
        "PLLfreq": PLL_freq,
        "P_vsc": P_vsc,
        "Q_vsc": Q_vsc,
        "I_vsc": I_vsc,
        "VSC_freq": VSC_freq,
        "VSC_rocof": VSC_rocof,
        "minV": float(np.min(V_mag)),
        "maxErr": float(np.max(np.abs(err))),
        "rmsErr": float(np.sqrt(np.mean(err**2))),
        "maxFreq": float(np.max(np.abs(PLL_freq))),
        "settle": settling,
        "minP": float(np.min(P_vsc)),
        "maxP": float(np.max(P_vsc)),
        "minQ": float(np.min(Q_vsc)),
        "maxQ": float(np.max(Q_vsc)),
        "maxI": float(np.max(I_vsc)),
    }


def print_metrics(r):
    ms = int(r["duration"] * 1000)

    print(f"\n--- Fault duration = {ms} ms ---")
    print(f"Min |V| at {PLL_BUS}:        {r['minV']:.4f} p.u.")
    print(f"Max |angle error|:          {r['maxErr']:.4f} rad")
    print(f"RMS angle error:            {r['rmsErr']:.4f} rad")
    print(f"Max |frequency deviation|:  {r['maxFreq']:.4f}")
    print(f"Settling time (<{ERR_THR}): {r['settle']:.4f} s")
    print(f"VSC P range:                {r['minP']:.4f} to {r['maxP']:.4f} p.u.")
    print(f"VSC Q range:                {r['minQ']:.4f} to {r['maxQ']:.4f} p.u.")
    print(f"Max |VSC current|:          {r['maxI']:.4f} p.u.")


# ============================
# MAIN
# ============================
if __name__ == "__main__":

    print("\n==============================================")
    print("PLL + VSC Fault Duration Sensitivity Study")
    print("==============================================")
    print(f"PLL bus     : {PLL_BUS}")
    print(f"Fault bus   : {FAULT_BUS}")
    print(f"Fault start : {T_FAULT_START} s")
    print(f"Durations   : {DURATIONS} s")
    print(f"T_filter    : {T_FILTER} s")
    print("==============================================\n")

    t0 = time.time()
    results = [run_case(d) for d in DURATIONS]
    print(f"All simulations completed in {time.time() - t0:.2f} s")

    for r in results:
        print_metrics(r)

    max_end = max(r["t_fault_end"] for r in results)

    # 1. PLL angle error
    plt.figure()
    for r in results:
        ms = int(r["duration"] * 1000)
        plt.plot(r["t"], r["err"], label=f"{ms} ms fault")
    plt.axvspan(T_FAULT_START, max_end, color="red", alpha=0.20)
    plt.title(f"PLL Angle Error vs Fault Duration")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle error (rad)")
    plt.grid(True)
    plt.legend()

    # 2. PLL frequency
    plt.figure()
    for r in results:
        t = r["t"]
        ms = int(r["duration"] * 1000)
        mask = (t > T_FAULT_START - 0.3) & (t < r["t_fault_end"] + 1.0)
        plt.plot(t[mask], r["PLLfreq"][mask], label=f"{ms} ms fault")
    plt.axvspan(T_FAULT_START, max_end, color="red", alpha=0.20)
    plt.title("External PLL Frequency Response")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency deviation")
    plt.grid(True)
    plt.legend()

    # 3. VSC active power
    plt.figure()
    for r in results:
        ms = int(r["duration"] * 1000)
        plt.plot(r["t"], r["P_vsc"], label=f"{ms} ms fault")
    plt.axvspan(T_FAULT_START, max_end, color="red", alpha=0.20)
    plt.title("VSC Active Power Response vs Fault Duration")
    plt.xlabel("Time (s)")
    plt.ylabel("P_vsc (p.u.)")
    plt.grid(True)
    plt.legend()

    # 4. VSC reactive power
    plt.figure()
    for r in results:
        ms = int(r["duration"] * 1000)
        plt.plot(r["t"], r["Q_vsc"], label=f"{ms} ms fault")
    plt.axvspan(T_FAULT_START, max_end, color="red", alpha=0.20)
    plt.title("VSC Reactive Power Response vs Fault Duration")
    plt.xlabel("Time (s)")
    plt.ylabel("Q_vsc (p.u.)")
    plt.grid(True)
    plt.legend()

    # 5. VSC current magnitude
    plt.figure()
    for r in results:
        ms = int(r["duration"] * 1000)
        plt.plot(r["t"], r["I_vsc"], label=f"{ms} ms fault")
    plt.axvspan(T_FAULT_START, max_end, color="red", alpha=0.20)
    plt.title("VSC Current Magnitude vs Fault Duration")
    plt.xlabel("Time (s)")
    plt.ylabel("|I_vsc| (p.u.)")
    plt.grid(True)
    plt.legend()

    # 6. VSC internal frequency estimate
    plt.figure()
    for r in results:
        t = r["t"]
        ms = int(r["duration"] * 1000)
        mask = (t > T_FAULT_START - 0.3) & (t < r["t_fault_end"] + 1.0)
        plt.plot(t[mask], r["VSC_freq"][mask], label=f"{ms} ms fault")
    plt.axvspan(T_FAULT_START, max_end, color="red", alpha=0.20)
    plt.title("VSC Internal Frequency Estimate vs Fault Duration")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency estimate (Hz)")
    plt.grid(True)
    plt.legend()

    plt.show()