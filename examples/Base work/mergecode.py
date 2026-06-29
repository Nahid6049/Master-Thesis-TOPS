def RjKvill_eqvivalent():
    return {
        'base_mva': 1000,
        'f': 50,
        'slack_bus': 'B1',

        'buses': [
            ['name','V_n'],
            ['B1',20],
            ['B2',420],
            ['B5',420],
            ['B6',420],
            ['B7',420],
            ['B8',340],
            ['B9',420],
            ['B10',20],
        ],

        'lines': [
            ['name','from_bus','to_bus','length','S_n','V_n','unit','R','X','B'],
            ['L2-5','B2','B5',70,1000,0,'p.u.',1e-4,1e-3,1.75e-3],
            ['L5-6','B5','B6',130,1000,0,'p.u.',1e-4,1e-3,1.75e-3],
            ['L6-7','B6','B7',10,1000,0,'p.u.',1e-4,1e-3,1.75e-3],
            ['L6-9','B6','B9',15,1000,0,'p.u.',1e-4,1e-3,1.75e-3],
        ],

        'transformers': [
            ['name','from_bus','to_bus','S_n','V_n_from','V_n_to','R','X'],
            ['T1','B1','B2',9000,20,420,0,0.15],
            ['T3','B7','B8',1500,420,340,0,0.15],
            ['T4','B9','B10',1500,420,20,0,0.15],
        ],

        'loads': [
            ['name','bus','P','Q','model'],
            ['L1','B5',3000,100,'Z'],
            ['L2','B6',800,50,'Z'],
        ],

        # ----------------------------
        # SHUNTS (COMMENTED OUT)
        # ----------------------------
        # 'shunts': {
        #     'Shunt': [
        #         ['name','bus','V_n','Q','model'],
        #         ['C1','B5',420,100,'Z'],
        #         ['C2','B6',420,200,'Z'],
        #     ],
        # },

     'generators': {
    'GEN': [
        ['name','bus','S_n','V_n','P','V','H','D',
         'X_d','X_q','X_d_t','X_q_t',
         'X_d_st','X_q_st',
         'T_d0_t','T_q0_t','T_d0_st','T_q0_st'],

        ['G1','B1',9000,20,3200,1.0,10.0,0.0,
         1.0,0.8,0.8,0.3,0.2,0.2,8.0,0.4,0.15,0.15],

        ['G3','B10',1440,20,300,1.0,4.0,0.0,
         1.02,0.63,0.25,0.63,0.16,0.16,6.5,1,0.05,0.15],

       
    ],
},

        # ----------------------------
        'vsc': {
            'VSC_SI': [
                ['name', 'bus', 'S_n', 'p_ref', 'q_ref',
                 'k_p', 'k_q', 'T_p', 'T_q',
                 'k_pll', 'T_pll', 'T_i',
                 'i_max', 'K_SI', 'T_SI', 'P_SI_max'],

                ['VSC1', 'B8',
                 1000,
                 0.2,
                 0.0,
                 1.0, 1.0,
                 0.1, 0.1,
                 5.0, 0.1,
                 0.05,
                 1.2,
                 0.0,
                 0.1,
                 1.2],
            ]
        },

        'gov': {
            # 'TGOV1': [
            #     ['name', 'gen', 'R', 'D_t', 'V_min', 'V_max', 'T_1', 'T_2', 'T_3'],
            #     ['GOV1', 'G1', 0.05, 0.02, 0, 1, 0.1 , 0.09, 0.2],
            #     #['GOV2', 'G2', 0.05, 0.02, 0, 1, 0.1 , 0.09, 0.2],
            #     ['GOV3', 'G3', 0.05, 0.02, 0, 1, 0.1 , 0.09, 0.2],
            # ],
             'HYGOV':[
                 ['name', 'gen', 'R', 'r' , 'T_f', 'T_r', 'T_g', 'A_t', 'T_w', 'q_nl' , 'D_turb', 'G_min', 'V_elm', 'G_max', 'P_N'],
                 ['HYGOV1', 'G1', 0.1, 1.5, 0.1, 2.0, 1, 1, 1, 0.01, 0.01, 0, 0.1, 1, 0],
                 ['HYGOV3', 'G3', 0.1, 1.5, 0.1, 2.0, 1, 1, 1, 0.01, 0.01, 0, 0.1, 1, 0],
             ],
        },

        'avr': {
            'SEXS': [
                ['name', 'gen', 'K', 'T_a', 'T_b', 'T_e', 'E_min', 'E_max'],
                ['AVR1', 'G1', 75, 2.0, 10.0, 0.5, 0, 6],
                # ['AVR2', 'G2', 600, 2.0, 10.0, 0.5, -3, 3],
                ['AVR3', 'G3', 100, 0.5, 3.0, 0.1, -3, 3],
            ],
        },

        'pss': {
            'STAB1': [
                ['name','gen','K','T','T_1','T_2','T_3','T_4','H_lim'],
                ['PSS1','G1',50,10.0,0.5,0.5,0.05,0.05,0.03],
                # ['PSS2','G2',50,10.0,0.5,0.5,0.05,0.05,0.03],
                ['PSS3','G3',50,10.0,0.5,0.5,0.05,0.02,0.03],
            ],
        },
    }

def load():
    return RjKvill_eqvivalent()

# simulate_gen.py
# -------------------------------------------------------
# Generator dynamic study (NO PLL)
# Fault at B8
# -------------------------------------------------------

from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import tops.dynamic as dps
import tops.solvers as dps_sol

import my_network


if __name__ == "__main__":

    # -------------------------------------------------------
    # Load base network (Generator model only)
    # -------------------------------------------------------
    model = my_network.load()

    # IMPORTANT:
    # Make sure no PLL is included in my_network.py
    # This file studies pure synchronous generator dynamics

    ps = dps.PowerSystemModel(model=model)

    # Power flow + initialization
    ps.power_flow()
    ps.init_dyn_sim()

    # Simulation setup
    t_end = 10.0
    x0 = ps.x0.copy()

    sol = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0.0, x0, t_end,
        max_step=5e-3
    )

    # -------------------------------------------------------
    # Fault at B8 (50 ms three-phase)
    # -------------------------------------------------------
    bus_names = list(ps.buses['name'])
    fault_bus = 'B8'
    k = bus_names.index(fault_bus)

    # Storage
    result_dict = defaultdict(list)
    P_e_stored = []
    E_f_stored = []
    V_stored = []

    t = 0.0
    t0 = time.time()

    # -------------------------------------------------------
    # Time simulation loop
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

        result_dict[('Global', 't')].append(t)

        for desc, state in zip(ps.state_desc, x):
            result_dict[tuple(desc)].append(state)

        P_e_stored.append(ps.gen['GEN'].P_e(x, v).copy())
        E_f_stored.append(ps.gen['GEN'].E_f(x, v).copy())
        V_stored.append(float(np.abs(v[k])))

    print(f"Generator simulation completed in {time.time() - t0:.2f} s")

    # -------------------------------------------------------
    # Post-processing
    # -------------------------------------------------------
    result = pd.DataFrame({kk: pd.Series(vv) for kk, vv in result_dict.items()})
    result.columns = pd.MultiIndex.from_tuples(result.columns)

    time_series = result[('Global', 't')].to_numpy()

    speed_df = result.xs(key='speed', axis='columns', level=1)
    angle_df = result.xs(key='angle', axis='columns', level=1)

    P_e = np.array(P_e_stored)
    E_f = np.array(E_f_stored)
    V_mag = np.array(V_stored)

    gen_rows = model['generators']['GEN'][1:]
    gen_names = [row[0] for row in gen_rows]
    gen_Sn = np.array([row[2] for row in gen_rows], dtype=float)

    # -------------------------------------------------------
    # Plots
    # -------------------------------------------------------
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
    plt.plot(time_series, E_f)
    plt.title("Field Voltage E_f")
    plt.xlabel("Time (s)")
    plt.ylabel("E_f (p.u.)")
    plt.legend([f"{g} E_f" for g in gen_names])

    plt.figure()
    plt.plot(time_series, V_mag)
    plt.title("Voltage Magnitude at B8")
    plt.xlabel("Time (s)")
    plt.ylabel("|V| (p.u.)")

    plt.show()
    
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


import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time

# ----------------------------
# Make Base work visible
# ----------------------------
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import tops.dynamic as dps
import tops.solvers as dps_sol
import my_network


# ============================
# SETTINGS
# ============================
PLL_BUS = "B8"
FAULT_BUS = "B8"          # change to "B2" if you want remote fault
T_FAULT_START = 1.0
T_END = 10.0
FAULT_ADMITTANCE = 10000.0

MAX_STEP = 5e-3
ERR_THR = 0.05  # rad (for settling time)

# Fault durations (seconds)
DURATIONS = [0.05, 0.10, 0.20]  # 50ms, 100ms, 200ms

# Keep PLL filter constant fixed (from your previous baseline)
T_FILTER = 0.05


# ============================
# RUN ONE CASE
# ============================
def run_case(duration: float):
    t_fault_end = T_FAULT_START + duration

    model = my_network.load()

    # Add PLL
    model["pll"] = {
        "PLL1": [
            ["name", "T_filter", "bus"],
            [f"PLL_{PLL_BUS}", float(T_FILTER), PLL_BUS],
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

    t = 0.0
    while t < T_END:

        # Apply fault
        if T_FAULT_START < t < t_fault_end:
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
        th_est = ps.pll["PLL1"].output(x, v)

        PLL_freq.append(float(np.array(f_est).flatten()[0]))
        PLL_ang.append(float(np.array(th_est).flatten()[0]))

    # Convert arrays
    t_store = np.array(t_store)
    V_mag = np.array(V_mag)
    V_ang = np.unwrap(np.array(V_ang))
    PLL_ang = np.unwrap(np.array(PLL_ang))
    PLL_freq = np.array(PLL_freq)

    err = PLL_ang - V_ang

    # Metrics
    min_voltage = float(np.min(V_mag))
    max_err = float(np.max(np.abs(err)))
    rms_err = float(np.sqrt(np.mean(err**2)))
    max_freq = float(np.max(np.abs(PLL_freq)))

    # Settling time after fault clear: error stays < ERR_THR
    settling = np.nan
    idx_post = np.where(t_store >= t_fault_end)[0]
    if len(idx_post) > 0:
        epost = np.abs(err[idx_post])
        for i in range(len(epost)):
            if np.all(epost[i:] < ERR_THR):
                settling = float(t_store[idx_post[i]] - t_fault_end)
                break

    return {
        "duration": float(duration),
        "t_fault_end": float(t_fault_end),
        "t": t_store,
        "Vmag": V_mag,
        "err": err,
        "PLLfreq": PLL_freq,
        "minV": min_voltage,
        "maxErr": max_err,
        "rmsErr": rms_err,
        "maxFreq": max_freq,
        "settle": settling,
    }


def print_metrics(res):
    ms = int(res["duration"] * 1000)
    print(f"\n--- Fault duration = {ms} ms ---")
    print(f"Min |V| at {PLL_BUS}:        {res['minV']:.4f} p.u.")
    print(f"Max |angle error|:          {res['maxErr']:.4f} rad")
    print(f"RMS angle error:            {res['rmsErr']:.4f} rad")
    print(f"Max |frequency deviation|:  {res['maxFreq']:.4f}")
    print(f"Settling time (<{ERR_THR}): {res['settle']:.4f} s")


# ============================
# MAIN
# ============================
if __name__ == "__main__":

    print("\n======================================")
    print("PLL Fault Duration Sensitivity Study")
    print("======================================")
    print(f"PLL bus   : {PLL_BUS}")
    print(f"Fault bus : {FAULT_BUS}")
    print(f"Fault start: {T_FAULT_START}s")
    print(f"Durations : {DURATIONS} s")
    print(f"T_filter  : {T_FILTER} s")
    print("======================================\n")

    t0 = time.time()
    results = [run_case(d) for d in DURATIONS]
    print(f"All simulations completed in {time.time() - t0:.2f} s")

    for r in results:
        print_metrics(r)

    # ============================
    # PLOTS
    # ============================

    # 1) Angle error overlay
    plt.figure()
    for r in results:
        ms = int(r["duration"] * 1000)
        plt.plot(r["t"], r["err"], label=f"{ms} ms fault")
    # mark widest fault window for shading (use max duration)
    max_end = max(rr["t_fault_end"] for rr in results)
    plt.axvspan(T_FAULT_START, max_end, color="red", alpha=0.20)
    plt.title(f"PLL Angle Error vs Fault Duration (PLL at {PLL_BUS}, Fault at {FAULT_BUS})")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle error (rad)")
    plt.grid(True)
    plt.legend()

    # 2) Frequency (zoom) overlay
    plt.figure()
    for r in results:
        t = r["t"]
        ms = int(r["duration"] * 1000)
        mask = (t > (T_FAULT_START - 0.3)) & (t < (r["t_fault_end"] + 1.0))
        plt.plot(t[mask], r["PLLfreq"][mask], label=f"{ms} ms fault")
    max_end = max(rr["t_fault_end"] for rr in results)
    plt.axvspan(T_FAULT_START, max_end, color="red", alpha=0.20)
    plt.title(f"PLL Frequency Response vs Fault Duration (Fault at {FAULT_BUS})")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency deviation")
    plt.grid(True)
    plt.legend()

    plt.show()


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
    plt.grid(True)
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
    plt.grid(True)
    plt.legend()

    plt.show()
    
    
    
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



# -*- coding: utf-8 -*-
"""
RjKvill Network
VSC_SI + PLL Interaction
Generator Electrical Power Added
PLL internal signals + corrected PLL angle oscillation
Fault at B8
"""

import sys
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import numpy as np
import tops.dynamic as dps
import tops.solvers as dps_sol
import importlib


# ==========================================================
# PATH FIX
# ==========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
base_work_dir = os.path.dirname(current_dir)
examples_dir = os.path.dirname(base_work_dir)
tops_root = os.path.dirname(examples_dir)

sys.path.append(base_work_dir)
sys.path.append(tops_root)


if __name__ == '__main__':

    # ==========================================================
    # LOAD NETWORK
    # ==========================================================
    import my_network as model_data
    importlib.reload(model_data)
    model = model_data.load()

    import examples.user_models.user_lib as user_lib


    # ==========================================================
    # INITIALIZE SYSTEM
    # ==========================================================
    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_lib)
    ps.init_dyn_sim()

    print("Initial mismatch:", max(abs(ps.ode_fun(0, ps.x_0))))


    # ==========================================================
    # SOLVER
    # ==========================================================
    t_end = 20

    sol = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0,
        ps.x_0.copy(),
        t_end,
        max_step=5e-3
    )


    # ==========================================================
    # STORAGE
    # ==========================================================
    t = 0
    res = defaultdict(list)
    t0 = time.time()

    idx = {'B1':0,'B2':1,'B5':2,'B6':3,'B7':4,'B8':5,'B9':6,'B10':7}

    iB6 = idx['B6']
    iB7 = idx['B7']
    iB8 = idx['B8']


    # ==========================================================
    # SIMULATION LOOP
    # ==========================================================
    while t < t_end:

        sys.stdout.write("\r%d%%" % (t/t_end*100))

        # Load change
        if t > 10:
            ps.y_bus_red_mod[iB6, iB6] = -0.1
        else:
            ps.y_bus_red_mod[iB6, iB6] = 0

        # Fault at B8
        if 2 < t < 2.1:
            ps.y_bus_red_mod[iB8, iB8] = 1000
        else:
            ps.y_bus_red_mod[iB8, iB8] = 0


        # Solver step
        x = sol.y
        sol.step()
        t = sol.t
        v = sol.v

        for mdl in ps.dyn_mdls:
            mdl.reset_outputs()


        # ======================================================
        # VSC INTERNAL STATES
        # ======================================================
        Xvsc = ps.vsc['VSC_SI'].local_view(x)

        pll_angle = float(np.squeeze(Xvsc['angle']))
        i_d = float(np.squeeze(Xvsc['i_d']))
        i_q = float(np.squeeze(Xvsc['i_q']))
        Vq = float(np.squeeze(ps.vsc['VSC_SI'].v_q(x, v)))


        # ======================================================
        # Generator speed
        # ======================================================
        SpeedVector = ps.gen['GEN'].speed(x, v)
        frequency = 50 + 50*np.mean(SpeedVector)


        # ======================================================
        # Generator electrical power
        # ======================================================
        P_gen = ps.gen['GEN'].P_e(x, v)


        # ======================================================
        # VSC power
        # ======================================================
        P_vsc = ps.vsc['VSC_SI'].p_e(x, v) * ps.sys_data['s_n']
        Q_vsc = ps.vsc['VSC_SI'].q_e(x, v) * ps.sys_data['s_n']


        # ======================================================
        # Line power
        # ======================================================
        I_line = ps.y_bus_red_full[iB6, iB7] * (v[iB6] - v[iB7])
        S_line = v[iB6] * np.conj(I_line) * ps.sys_data['s_n']


        # ======================================================
        # Bus voltage
        # ======================================================
        V_B8 = np.abs(v[iB8])


        # ======================================================
        # STORE RESULTS
        # ======================================================
        res['t'].append(t)
        res['gen_speed'].append(SpeedVector.copy())
        res['P_gen'].append(P_gen.copy())
        res['frequency'].append(frequency)
        res['P_vsc'].append(P_vsc)
        res['Q_vsc'].append(Q_vsc)
        res['P_line'].append(np.real(S_line))
        res['V_B8'].append(V_B8)

        res['pll_angle'].append(pll_angle)
        res['Vq'].append(Vq)
        res['i_d'].append(i_d)
        res['i_q'].append(i_q)


    print("\nSimulation completed in {:.2f} seconds".format(time.time()-t0))


    # ==========================================================
    # POST PROCESSING
    # ==========================================================
    t_vec = np.array(res['t'])

    genspeed1 = [res['gen_speed'][i][0] for i in range(len(res['t']))]
    genspeed2 = [res['gen_speed'][i][1] for i in range(len(res['t']))]

    P_g1 = [res['P_gen'][i][0] for i in range(len(res['t']))]
    P_g2 = [res['P_gen'][i][1] for i in range(len(res['t']))]


    # ==========================================================
    # PLL ANGLE PROCESSING
    # ==========================================================
    pll_angle = np.unwrap(np.array(res['pll_angle']))
    pll_angle = pll_angle - pll_angle[0]

    m, b = np.polyfit(t_vec, pll_angle, 1)
    theta_osc = pll_angle - (m*t_vec + b)


    # ==========================================================
    # PLOTS
    # ==========================================================

    plt.figure()
    plt.plot(t_vec, genspeed1, color='green', label='GEN1')
    plt.plot(t_vec, genspeed2, color='orange', label='GEN2')
    plt.xlabel('Time [s]')
    plt.ylabel('Speed deviation [pu]')
    plt.legend()
    plt.title("Generator Speed Response")


    plt.figure()
    plt.plot(t_vec, P_g1, color='green', label='GEN1')
    plt.plot(t_vec, P_g2, color='orange', label='GEN2')
    plt.xlabel('Time [s]')
    plt.ylabel('Generator Electrical Power [MW]')
    plt.legend()
    plt.title("Generator Electrical Power Response")


    plt.figure()
    plt.plot(t_vec, res['frequency'], color='green')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.title("System Frequency Response")


    plt.figure()
    plt.plot(t_vec, res['P_vsc'], color='green')
    plt.xlabel('Time [s]')
    plt.ylabel('VSC Active Power [MW]')
    plt.title("VSC Power Oscillation")

    plt.figure()
    plt.plot(t_vec, res['Q_vsc'], color='orange')
    plt.xlabel('Time [s]')
    plt.ylabel('VSC Reactive Power [MW]')
    plt.title("VSC Reactive Power Response")



    plt.figure()
    plt.plot(t_vec, res['Vq'], color='green')
    plt.xlabel('Time [s]')
    plt.ylabel('Vq (PLL Phase Error)')
    plt.title("PLL Phase Error")


    plt.figure()
    plt.plot(t_vec, res['i_d'], color='green', label='i_d')
    plt.plot(t_vec, res['i_q'], color='orange', label='i_q')
    plt.xlabel('Time [s]')
    plt.ylabel('Current [pu]')
    plt.legend()
    plt.title("dq Current Components")


    # ==========================================================
    # PLL–VSC INTERACTION PLOT
    # ==========================================================
    fig, ax = plt.subplots(3,1,sharex=True)

    ax[0].plot(t_vec, res['Vq'], color='green')
    ax[0].axvline(2, color='orange')
    ax[0].axvline(10, color='orange')
    ax[0].set_ylabel("Vq")
    ax[0].set_title("PLL–VSC Interaction")


    ax[1].plot(t_vec, res['i_d'], color='green', label='i_d')
    ax[1].plot(t_vec, res['i_q'], color='orange', label='i_q')
    ax[1].axvline(2, color='orange')
    ax[1].axvline(10, color='orange')
    ax[1].set_ylabel("Current [pu]")
    ax[1].legend()


    ax[2].plot(t_vec, res['P_vsc'], color='green', label='P_vsc')
    ax[2].plot(t_vec, res['Q_vsc'], color='orange', label='Q_vsc')
    ax[2].axvline(2, color='orange')
    ax[2].axvline(10, color='orange')
    ax[2].set_ylabel("Power [MW]")
    ax[2].set_xlabel("Time [s]")
    ax[2].legend()


    plt.tight_layout()
    plt.show()

    # -*- coding: utf-8 -*-
"""
PLL Sensitivity + Finish PLL–VSC Interaction
- Runs 3 cases: T_pll = 0.02, 0.1, 0.2
- Disturbances: Fault at B8 (2–2.1 s), Load increase at B6 (t>10 s)
- Outputs added:
    (1) PLL freq estimate vs system freq
    (2) PLL angle tracking vs voltage angle
    (3) Angle error (wrapped)
    (4) dq voltages (Vd, Vq)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import importlib

import tops.dynamic as dps
import tops.solvers as dps_sol

# ---------------------------------------------------------
# PATH FIX (same pattern you used)
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
base_work_dir = os.path.dirname(current_dir)
examples_dir = os.path.dirname(base_work_dir)
tops_root = os.path.dirname(examples_dir)

sys.path.append(base_work_dir)
sys.path.append(tops_root)

import examples.user_models.user_lib as user_lib


# ---------------------------------------------------------
# PLL CASES
# ---------------------------------------------------------
pll_cases = [0.02, 0.1, 0.2]
labels = ['Tpll = 0.02', 'Tpll = 0.1', 'Tpll = 0.2']

# No blue: green / orange / red
colors = ['green', 'orange', 'red']
linestyles = ['-', '--', '-.']
LW = 1  # thinner lines


# ---------------------------------------------------------
# Utility: wrap angle to [-pi, pi]
# ---------------------------------------------------------
def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------
# SIMULATION FUNCTION
# ---------------------------------------------------------
def run_simulation(T_pll_value, t_end=20.0, max_step=5e-3):

    import my_network as model_data
    importlib.reload(model_data)
    model = model_data.load()

    # Update T_pll in model (your VSC_SI table index [1][10] is T_pll)
    model['vsc']['VSC_SI'][1][10] = T_pll_value

    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_lib)
    ps.init_dyn_sim()

    sol = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0,
        ps.x_0.copy(),
        t_end,
        max_step=max_step
    )

    # Bus mapping (your reduced order)
    idx = {'B1': 0, 'B2': 1, 'B5': 2, 'B6': 3, 'B7': 4, 'B8': 5, 'B9': 6, 'B10': 7}
    iB6 = idx['B6']
    iB8 = idx['B8']

    res = defaultdict(list)
    t = 0.0

    while t < t_end:

        # -------------------------
        # Disturbances
        # -------------------------
        # Load change at B6 after 10s
        if t > 10:
            ps.y_bus_red_mod[iB6, iB6] = -0.1
        else:
            ps.y_bus_red_mod[iB6, iB6] = 0

        # Fault at B8 (2–2.1s)
        if 2 < t < 2.1:
            ps.y_bus_red_mod[iB8, iB8] = 1000
        else:
            ps.y_bus_red_mod[iB8, iB8] = 0

        # -------------------------
        # Step solver
        # -------------------------
        x = sol.y
        sol.step()
        t = sol.t
        v = sol.v

        for mdl in ps.dyn_mdls:
            mdl.reset_outputs()

        # -------------------------
        # Extract signals
        # -------------------------
        Xvsc = ps.vsc['VSC_SI'].local_view(x)

        # PLL angle (state)
        theta_pll = float(np.squeeze(Xvsc['angle']))

        # Voltage angle at B8 (actual angle PLL tries to track)
        theta_v = float(np.angle(v[iB8]))

        # PLL dq voltage components:
        # v_dq = v_t * exp(-j*theta_pll)
        v_t = v[iB8]
        v_dq = v_t * np.exp(-1j * theta_pll)
        Vd = float(np.real(v_dq))
        Vq = float(np.imag(v_dq))  # should match v_q method

        # PLL frequency estimate (Hz) from your model method
        f_pll = float(np.squeeze(ps.vsc['VSC_SI'].freq_est(x, v)))

        # VSC powers
        P_vsc = float(np.squeeze(ps.vsc['VSC_SI'].p_e(x, v) * ps.sys_data['s_n']))
        Q_vsc = float(np.squeeze(ps.vsc['VSC_SI'].q_e(x, v) * ps.sys_data['s_n']))

        # System frequency from generators (your previous approach)
        SpeedVector = ps.gen['GEN'].speed(x, v)
        f_sys = float(50 + 50 * np.mean(SpeedVector))

        # dq currents (states)
        i_d = float(np.squeeze(Xvsc['i_d']))
        i_q = float(np.squeeze(Xvsc['i_q']))

        # -------------------------
        # Store
        # -------------------------
        res['t'].append(t)

        res['theta_pll'].append(theta_pll)
        res['theta_v'].append(theta_v)

        res['Vd'].append(Vd)
        res['Vq'].append(Vq)

        res['f_pll'].append(f_pll)
        res['f_sys'].append(f_sys)

        res['i_d'].append(i_d)
        res['i_q'].append(i_q)

        res['P_vsc'].append(P_vsc)
        res['Q_vsc'].append(Q_vsc)

    return res


# ---------------------------------------------------------
# RUN ALL CASES
# ---------------------------------------------------------
results = []
for Tpll in pll_cases:
    print(f"Running simulation with T_pll = {Tpll}")
    results.append(run_simulation(Tpll))


# ---------------------------------------------------------
# PLOTTING HELPERS
# ---------------------------------------------------------
def plot_multi(title, y_key, y_label, xlim=None):
    plt.figure()
    for i in range(len(results)):
        t = np.array(results[i]['t'])
        y = np.array(results[i][y_key])
        plt.plot(
            t, y,
            color=colors[i],
            linestyle=linestyles[i],
            linewidth=LW,
            label=labels[i]
        )
    plt.xlabel('Time [s]')
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    if xlim is not None:
        plt.xlim(xlim)


# ---------------------------------------------------------
# (1) PLL phase error (Vq)  ✅
# ---------------------------------------------------------
plot_multi('PLL Sensitivity: Phase Error (Vq)', 'Vq', 'Vq (pu)')

# Optional zoom near fault
plot_multi('PLL Phase Error (Zoom near fault)', 'Vq', 'Vq (pu)', xlim=(1.9, 2.6))


# ---------------------------------------------------------
# (2) PLL frequency estimate vs System frequency  ✅
# ---------------------------------------------------------
plt.figure()
for i in range(len(results)):
    t = np.array(results[i]['t'])
    f_sys = np.array(results[i]['f_sys'])
    f_pll = np.array(results[i]['f_pll'])

    # plot system freq (solid) and pll freq (dashed) using same case color
    plt.plot(t, f_sys, color=colors[i], linestyle='-', linewidth=LW, label=f'{labels[i]}: f_sys')
    plt.plot(t, f_pll, color=colors[i], linestyle='--', linewidth=LW, label=f'{labels[i]}: f_PLL')

plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.title('PLL Frequency Estimate vs System Frequency')
plt.legend(ncol=2)
plt.grid(True, alpha=0.3)


# ---------------------------------------------------------
# (3) PLL angle tracking vs voltage angle  ✅
# ---------------------------------------------------------
plt.figure()
for i in range(len(results)):
    t = np.array(results[i]['t'])

    theta_pll = np.unwrap(np.array(results[i]['theta_pll']))
    theta_v = np.unwrap(np.array(results[i]['theta_v']))

    # relative angles (start at 0)
    theta_pll = theta_pll - theta_pll[0]
    theta_v = theta_v - theta_v[0]

    plt.plot(t, theta_v, color=colors[i], linestyle='-', linewidth=LW, label=f'{labels[i]}: θ_V')
    plt.plot(t, theta_pll, color=colors[i], linestyle='--', linewidth=LW, label=f'{labels[i]}: θ_PLL')

plt.xlabel('Time [s]')
plt.ylabel('Angle [rad]')
plt.title('PLL Angle Tracking vs Voltage Angle (B8)')
plt.legend(ncol=2)
plt.grid(True, alpha=0.3)


# ---------------------------------------------------------
# (4) Angle error (wrapped to [-pi, pi])  ✅
# ---------------------------------------------------------
plt.figure()
for i in range(len(results)):
    t = np.array(results[i]['t'])

    theta_pll = np.unwrap(np.array(results[i]['theta_pll']))
    theta_v = np.unwrap(np.array(results[i]['theta_v']))

    # make them relative first (removes initial offset)
    theta_pll = theta_pll - theta_pll[0]
    theta_v = theta_v - theta_v[0]

    err = wrap_to_pi(theta_pll - theta_v)

    plt.plot(t, err, color=colors[i], linestyle=linestyles[i], linewidth=LW, label=labels[i])

plt.xlabel('Time [s]')
plt.ylabel('Angle error [rad]')
plt.title('PLL Angle Error (wrapped to [-π, π])')
plt.legend()
plt.grid(True, alpha=0.3)


# ---------------------------------------------------------
# (5) dq currents (id, iq)  ✅
# ---------------------------------------------------------
plot_multi('PLL Sensitivity: d-axis current i_d', 'i_d', 'i_d (pu)')
plot_multi('PLL Sensitivity: q-axis current i_q', 'i_q', 'i_q (pu)')

# Zoom near fault for currents
plot_multi('i_d (Zoom near fault)', 'i_d', 'i_d (pu)', xlim=(1.9, 2.6))
plot_multi('i_q (Zoom near fault)', 'i_q', 'i_q (pu)', xlim=(1.9, 2.6))


# ---------------------------------------------------------
# (6) VSC P and Q  ✅
# ---------------------------------------------------------
plot_multi('PLL Sensitivity: VSC Active Power P_vsc', 'P_vsc', 'P_vsc [MW]')
plot_multi('PLL Sensitivity: VSC Reactive Power Q_vsc', 'Q_vsc', 'Q_vsc [MW]')

# Zoom near fault for power
plot_multi('P_vsc (Zoom near fault)', 'P_vsc', 'P_vsc [MW]', xlim=(1.9, 2.6))
plot_multi('Q_vsc (Zoom near fault)', 'Q_vsc', 'Q_vsc [MW]', xlim=(1.9, 2.6))


# ---------------------------------------------------------
# (7) dq voltages Vd, Vq (optional but nice) ✅
# ---------------------------------------------------------
plot_multi('PLL dq Voltage: Vd', 'Vd', 'Vd (pu)')
plot_multi('PLL dq Voltage: Vq', 'Vq', 'Vq (pu)')

plt.show()


# -*- coding: utf-8 -*-
"""
RjKvill System
VSC_SI + PLL Study with Multiple Cases
"""

import sys
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import numpy as np
import tops.dynamic as dps
import tops.solvers as dps_sol
import importlib


# ==========================================================
# PATH FIX (YOUR ORIGINAL STRUCTURE)
# ==========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
base_work_dir = os.path.dirname(current_dir)
examples_dir = os.path.dirname(base_work_dir)
tops_root = os.path.dirname(examples_dir)

sys.path.append(base_work_dir)
sys.path.append(tops_root)


# ==========================================================
# SELECT STUDY CASE
# ==========================================================
CASE = 2   # Change 1,2,3,4


if __name__ == '__main__':

    # ==========================================================
    # LOAD NETWORK
    # ==========================================================
    import my_network as model_data
    importlib.reload(model_data)
    model = model_data.load()

    import examples.user_models.user_lib as user_lib

    # ==========================================================
    # PLL SETTINGS (Case-dependent)
    # ==========================================================
    if CASE == 4:
        pll_T = 0.02   # Faster PLL
    else:
        pll_T = 0.1    # Default PLL

    model['pll'] = {'PLL1': [
        ['name', 'T_filter', 'bus'],
        ['PLL1', pll_T, 'B8'],
    ]}

    # ==========================================================
    # INITIALIZE SYSTEM
    # ==========================================================
    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_lib)
    ps.init_dyn_sim()

    print("Running CASE:", CASE)
    print("Initial mismatch:",
          max(abs(ps.ode_fun(0, ps.x_0))))

    # ==========================================================
    # SOLVER
    # ==========================================================
    t_end = 20
    sol = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0,
        ps.x_0.copy(),
        t_end,
        max_step=5e-3
    )

    # ==========================================================
    # STORAGE
    # ==========================================================
    t = 0
    res = defaultdict(list)
    t0 = time.time()

    idx = {'B1':0,'B2':1,'B5':2,'B6':3,
           'B7':4,'B8':5,'B9':6,'B10':7}

    iB6 = idx['B6']
    iB7 = idx['B7']
    iB8 = idx['B8']

    # ==========================================================
    # SIMULATION LOOP
    # ==========================================================
    while t < t_end:

        sys.stdout.write("\r%d%%" % (t/t_end*100))

        # -------------------------
        # LOAD STEP (Case-dependent)
        # -------------------------
        if t > 10:
            if CASE == 3:
                ps.y_bus_red_mod[iB6, iB6] = -0.2   # Larger load
            else:
                ps.y_bus_red_mod[iB6, iB6] = -0.1   # Default load
        else:
            ps.y_bus_red_mod[iB6, iB6] = 0

        # -------------------------
        # FAULT LOCATION (Case-dependent)
        # -------------------------
        if 2 < t < 2.1:
            if CASE == 2:
                ps.y_bus_red_mod[iB8, iB8] = 1000   # Fault at B8
            else:
                ps.y_bus_red_mod[iB7, iB7] = 1000   # Default fault B7
        else:
            ps.y_bus_red_mod[iB7, iB7] = 0
            ps.y_bus_red_mod[iB8, iB8] = 0

        # -------------------------
        # SOLVER STEP
        # -------------------------
        x = sol.y
        sol.step()
        t = sol.t
        v = sol.v

        for mdl in ps.dyn_mdls:
            mdl.reset_outputs()

        # -------------------------
        # MEASUREMENTS
        # -------------------------
        SpeedVector = ps.gen['GEN'].speed(x, v)
        frequency = 50 + 50*np.mean(SpeedVector)

        P_gen = ps.gen['GEN'].P_e(x, v)

        pll_freq = ps.pll['PLL1'].freq_est(x, v)

        P_vsc = ps.vsc['VSC_SI'].p_e(x, v) * ps.sys_data['s_n']
        Q_vsc = ps.vsc['VSC_SI'].q_e(x, v) * ps.sys_data['s_n']

        I_line = ps.y_bus_red_full[iB6, iB7] * (v[iB6] - v[iB7])
        S_line = v[iB6] * np.conj(I_line) * ps.sys_data['s_n']

        V_B8 = np.abs(v[iB8])

        # -------------------------
        # STORE
        # -------------------------
        res['t'].append(t)
        res['gen_speed'].append(SpeedVector.copy())
        res['P_gen'].append(P_gen.copy())
        res['frequency'].append(frequency)
        res['pll_freq'].append(pll_freq.copy())
        res['P_vsc'].append(P_vsc)
        res['Q_vsc'].append(Q_vsc)
        res['P_line'].append(np.real(S_line))
        res['V_B8'].append(V_B8)

    print("\nSimulation completed in {:.2f} seconds"
          .format(time.time()-t0))

    # ==========================================================
    # POST PROCESSING
    # ==========================================================
    genspeed1 = [res['gen_speed'][i][0] for i in range(len(res['t']))]
    genspeed2 = [res['gen_speed'][i][1] for i in range(len(res['t']))]

    P_g1 = [res['P_gen'][i][0] for i in range(len(res['t']))]
    P_g2 = [res['P_gen'][i][1] for i in range(len(res['t']))]

    # ==========================================================
    # PLOTS
    # ==========================================================
    plt.figure()
    plt.plot(res['t'], genspeed1, label='GEN1')
    plt.plot(res['t'], genspeed2, label='GEN2')
    plt.title("Generator Speed")
    plt.legend()

    plt.figure()
    plt.plot(res['t'], P_g1, label='GEN1')
    plt.plot(res['t'], P_g2, label='GEN2')
    plt.title("Generator Electrical Power")
    plt.legend()

    plt.figure()
    plt.plot(res['t'], res['frequency'])
    plt.title("System Frequency")

    plt.figure()
    plt.plot(res['t'], 50+50*np.array(res['pll_freq']))
    plt.title("PLL Frequency")

    plt.figure()
    plt.plot(res['t'], res['P_vsc'])
    plt.title("VSC Active Power")

    plt.figure()
    plt.plot(res['t'], res['Q_vsc'])
    plt.title("VSC Reactive Power")

    plt.figure()
    plt.plot(res['t'], res['P_line'])
    plt.title("Line Power Transfer")

    plt.figure()
    plt.plot(res['t'], res['V_B8'])
    plt.title("Voltage at B8")

    plt.show()


    # -*- coding: utf-8 -*-
"""
Synthetic Inertia Sensitivity Study

Cases:
K_SI = 0
K_SI = 0.5
K_SI = 1.0

Disturbances
- Fault at B8 (2–2.1 s)
- Load change at B6 (after 10 s)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import importlib

import tops.dynamic as dps
import tops.solvers as dps_sol


# ---------------------------------------------------------
# PATH FIX
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
base_work_dir = os.path.dirname(current_dir)
examples_dir = os.path.dirname(base_work_dir)
tops_root = os.path.dirname(examples_dir)

sys.path.append(base_work_dir)
sys.path.append(tops_root)

import examples.user_models.user_lib as user_lib


# ---------------------------------------------------------
# CASES
# ---------------------------------------------------------
K_cases = [0, 0.5, 1.0]
labels = ['K_SI = 0', 'K_SI = 0.3', 'K_SI = 0.5']
colors = ['green', 'orange', 'red']


# ---------------------------------------------------------
# SIMULATION FUNCTION
# ---------------------------------------------------------
def run_simulation(K_SI_value):

    import my_network as model_data
    importlib.reload(model_data)
    model = model_data.load()

    # modify inertia gain
    model['vsc']['VSC_SI'][1][13] = K_SI_value

    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_lib)
    ps.init_dyn_sim()

    sol = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0,
        ps.x_0.copy(),
        20,
        max_step=5e-3
    )

    idx = {'B1':0,'B2':1,'B5':2,'B6':3,'B7':4,'B8':5,'B9':6,'B10':7}

    iB6 = idx['B6']
    iB8 = idx['B8']

    res = defaultdict(list)

    t = 0

    while t < 20:

        # Load change
        if t > 10:
            ps.y_bus_red_mod[iB6, iB6] = -0.1
        else:
            ps.y_bus_red_mod[iB6, iB6] = 0

        # Fault at B8
        if 2 < t < 2.1:
            ps.y_bus_red_mod[iB8, iB8] = 1000
        else:
            ps.y_bus_red_mod[iB8, iB8] = 0

        x = sol.y
        sol.step()
        t = sol.t
        v = sol.v

        for mdl in ps.dyn_mdls:
            mdl.reset_outputs()

        # System frequency
        SpeedVector = ps.gen['GEN'].speed(x, v)
        freq = 50 + 50*np.mean(SpeedVector)

        # VSC power
        P_vsc = ps.vsc['VSC_SI'].p_e(x, v) * ps.sys_data['s_n']

        # ROCOF
        rocof = ps.vsc['VSC_SI'].rocof_est(x, v)

        res['t'].append(t)
        res['freq'].append(freq)
        res['P_vsc'].append(P_vsc)
        res['rocof'].append(rocof)

    return res


# ---------------------------------------------------------
# RUN ALL CASES
# ---------------------------------------------------------
results = []

for K in K_cases:
    print("Running simulation K_SI =", K)
    results.append(run_simulation(K))


# ---------------------------------------------------------
# PLOTS
# ---------------------------------------------------------

# Frequency
plt.figure()

for i in range(3):
    t = np.array(results[i]['t'])
    f = np.array(results[i]['freq'])

    plt.plot(t, f, color=colors[i], label=labels[i], linewidth=1)

plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.title('System Frequency Response')
plt.legend()
plt.grid()


# ROCOF
plt.figure()

for i in range(3):
    t = np.array(results[i]['t'])
    r = np.array(results[i]['rocof'])

    plt.plot(t, r, color=colors[i], label=labels[i], linewidth=1)

plt.xlabel('Time [s]')
plt.ylabel('ROCOF [Hz/s]')
plt.title('ROCOF Response')
plt.legend()
plt.grid()


# VSC Active Power
plt.figure()

for i in range(3):
    t = np.array(results[i]['t'])
    P = np.array(results[i]['P_vsc'])

    plt.plot(t, P, color=colors[i], label=labels[i], linewidth=1)

plt.xlabel('Time [s]')
plt.ylabel('VSC Active Power [MW]')
plt.title('Synthetic Inertia Power Injection')
plt.legend()
plt.grid()


plt.show()

Above code is for system analysis how everything goes on with system...
after doing all of above things, Then i updated my network data..i added more generator in the local
area and perform SCR analysis, export import balance which is called power sharing and CCT analysis for various case..check the below code 

def RjKvill_eqvivalent():

    return {
        'base_mva': 1000,
        'f': 50,
        'slack_bus': 'B1',

        # ----------------------------
        # BUSES
        # ----------------------------
        'buses': [
            ['name','V_n'],
            ['B1',20],
            ['B2',420],
            ['B5',420],
            ['B6',420],
            ['B7',420],
            ['B8',340],
            ['B9',420],
            ['B10',20],
        ],

        # ----------------------------
        # LINES
        # ----------------------------
        'lines': [
            ['name','from_bus','to_bus','length','S_n','V_n','unit','R','X','B'],
            ['L2-5','B2','B5',70,1000,0,'p.u.',1e-4,1e-3,1.75e-3],
            ['L5-6','B5','B6',130,1000,0,'p.u.',1e-4,1e-3,1.75e-3],
            ['L6-7','B6','B7',10,1000,0,'p.u.',1e-4,1e-3,1.75e-3],
            ['L6-9','B6','B9',15,1000,0,'p.u.',1e-4,1e-3,1.75e-3],
        ],

        # ----------------------------
        # TRANSFORMERS
        # ----------------------------
        'transformers': [
            ['name','from_bus','to_bus','S_n','V_n_from','V_n_to','R','X'],
            ['T1','B1','B2',9000,20,420,0,0.15],
            ['T3','B7','B8',1500,420,340,0,0.15],
            ['T4','B9','B10',1500,420,20,0,0.15],
          
        ],

        # ----------------------------
        # LOADS
        # ----------------------------
        'loads': [
            ['name','bus','P','Q','model'],
            ['L1','B5',3000,100,'Z'],
            ['L2','B6',800,50,'Z'],
        ],

        # ----------------------------
        # GENERATORS (G3 DUPLICATED)
        # ----------------------------
        'generators': {
            'GEN': [
                ['name','bus','S_n','V_n','P','V','H','D',
                 'X_d','X_q','X_d_t','X_q_t',
                 'X_d_st','X_q_st',
                 'T_d0_t','T_q0_t','T_d0_st','T_q0_st'],

                ['G1','B1',9000,20,3200,1.0,10.0,0.0,
                 1.0,0.8,0.8,0.3,0.2,0.2,8.0,0.4,0.15,0.15],

                ['G3','B10',1440,20,300,1.0,4.0,0.0,
                 1.02,0.63,0.25,0.63,0.16,0.16,6.5,1,0.05,0.15],

                ['G4','B10',1440,20,0,1.0,4.0,0.0,
                 1.02,0.63,0.25,0.63,0.16,0.16,6.5,1,0.05,0.15],

                ['G5','B10',1440,20,0,1.0,4.0,0.0,
                 1.02,0.63,0.25,0.63,0.16,0.16,6.5,1,0.05,0.15],

                ['G6','B10',1440,20,0,1.0,4.0,0.0,
                 1.02,0.63,0.25,0.63,0.16,0.16,6.5,1,0.05,0.15],
            ],
        },

        # ----------------------------
        # VSC (UNCHANGED)
        # ----------------------------
        'vsc': {
            'VSC_SI': [
                ['name', 'bus', 'S_n', 'p_ref', 'q_ref',
                 'k_p', 'k_q', 'T_p', 'T_q',
                 'k_pll', 'T_pll', 'T_i',
                 'i_max', 'K_SI', 'T_SI', 'P_SI_max'],

                ['VSC1', 'B8',
                 1000,
                 0.2,
                 0.0,
                 1.0, 1.0,
                 0.1, 0.1,
                 5.0, 0.1,
                 0.05,
                 1.2,
                 0.0,
                 0.1,
                 1.2],
            ]
        },

        # ----------------------------
        # GOVERNOR (HYGOV FOR ALL)
        # ----------------------------
        'gov': {
            'HYGOV': [
                ['name','gen','R','r','T_f','T_r','T_g','A_t','T_w','q_nl','D_turb','G_min','V_elm','G_max','P_N'],

                ['HYGOV1','G1',0.1,1.5,0.1,2.0,1,1,1,0.01,0.01,0,0.1,1,0],

                ['HYGOV3','G3',0.1,1.5,0.1,2.0,1,1,1,0.01,0.01,0,0.1,1,0],
                ['HYGOV4','G4',0.1,1.5,0.1,2.0,1,1,1,0.01,0.01,0,0.1,1,0],
                ['HYGOV5','G5',0.1,1.5,0.1,2.0,1,1,1,0.01,0.01,0,0.1,1,0],
                ['HYGOV6','G6',0.1,1.5,0.1,2.0,1,1,1,0.01,0.01,0,0.1,1,0],
            ],
        },

        # ----------------------------
        # AVR
        # ----------------------------
        'avr': {
            'SEXS': [
                ['name','gen','K','T_a','T_b','T_e','E_min','E_max'],

                ['AVR1','G1',75,2.0,10.0,0.5,0,6],

                ['AVR3','G3',100,0.5,3.0,0.1,-3,3],
                ['AVR4','G4',100,0.5,3.0,0.1,-3,3],
                ['AVR5','G5',100,0.5,3.0,0.1,-3,3],
                ['AVR6','G6',100,0.5,3.0,0.1,-3,3],
            ],
        },

        # ----------------------------
        # PSS
        # ----------------------------
        'pss': {
            'STAB1': [
                ['name','gen','K','T','T_1','T_2','T_3','T_4','H_lim'],

                ['PSS1','G1',50,10.0,0.5,0.5,0.05,0.05,0.03],

                ['PSS3','G3',50,10.0,0.5,0.5,0.05,0.02,0.03],
                ['PSS4','G4',50,10.0,0.5,0.5,0.05,0.02,0.03],
                ['PSS5','G5',50,10.0,0.5,0.5,0.05,0.02,0.03],
                ['PSS6','G6',50,10.0,0.5,0.5,0.05,0.02,0.03],
            ],
        },
    }


def load():
    return RjKvill_eqvivalent()


    # -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
import importlib
import copy
from collections import defaultdict

import tops.dynamic as dps
import tops.solvers as dps_sol

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_PATH = r"D:\Masters REM+\Master Thesis\paper\TOPS-main\TOPS-main"

sys.path.insert(0, BASE_PATH + r"\examples\Base work")
sys.path.append(BASE_PATH + r"\examples\user_models")

import user_lib
import generator_network as model_data


# --------------------------------------------------
# SCR CALCULATION
# --------------------------------------------------
def compute_scr(model, n):

    Z_L56 = Z_L25 = Z_L69 = None
    Z_T1 = Z_T4 = None
    Z_G1 = None
    Z_G3 = None

    # Lines
    for line in model['lines'][1:]:
        Z = line[7] + 1j * line[8]
        if line[0] == 'L5-6': Z_L56 = Z
        elif line[0] == 'L2-5': Z_L25 = Z
        elif line[0] == 'L6-9': Z_L69 = Z

    # Transformers
    for tr in model['transformers'][1:]:
        Z = tr[6] + 1j * tr[7]
        if tr[0] == 'T1': Z_T1 = Z
        elif tr[0] == 'T4': Z_T4 = Z

    # Generators
    for gen in model['generators']['GEN'][1:]:
        if gen[0] == 'G1':
            Z_G1 = 1j * gen[12]   # Xd''
        if gen[0] == 'G3':
            Z_G3 = 1j * gen[12]

    Z_grid = Z_L25 + Z_L56 + Z_T1 + Z_G1
    Z_local = Z_L69 + Z_T4 + Z_G3

    if n == 0:
        Z_th = Z_grid
    else:
        Z_local_eq = Z_local / n
        Z_th = 1 / (1 / Z_grid + 1 / Z_local_eq)

    return 1 / abs(Z_th)


# --------------------------------------------------
# APPLY GENERATOR CASE + TRANSFORMER FIX
# --------------------------------------------------
def apply_generator_case(model, n):

    model = copy.deepcopy(model)

    # Select generators
    if n == 0:
        keep = {'G1'}
    elif n == 1:
        keep = {'G1', 'G3'}
    elif n == 2:
        keep = {'G1', 'G3', 'G4'}
    elif n == 4:
        keep = {'G1', 'G3', 'G4', 'G5', 'G6'}

    # Generators
    header = model['generators']['GEN'][0]
    model['generators']['GEN'] = [header] + [
        g for g in model['generators']['GEN'][1:] if g[0] in keep
    ]

    # Gov / AVR / PSS
    for key in ['gov', 'avr', 'pss']:
        subkey = list(model[key].keys())[0]
        header = model[key][subkey][0]
        model[key][subkey] = [header] + [
            g for g in model[key][subkey][1:] if g[1] in keep
        ]

    # 🔴 Transformer scaling (ONLY T4)
    for tr in model['transformers'][1:]:
        if tr[0] == 'T4':
            if n == 0:
                tr[3] = 360
            elif n == 1:
                tr[3] = 360
            elif n == 2:
                tr[3] = 720
            elif n == 4:
                tr[3] = 1440

    return model


# --------------------------------------------------
# SIMULATION
# --------------------------------------------------
def run_simulation(model):

    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_lib)

    ps.power_flow()
    ps.init_dyn_sim()

    sol = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0,
        ps.x_0.copy(),
        10,
        max_step=5e-3
    )

    idx = {'B1':0,'B2':1,'B5':2,'B6':3,'B7':4,'B8':5,'B9':6,'B10':7}
    iB6 = idx['B6']

    res = defaultdict(list)
    t = 0.0

    while t < 10:

        sol.step()

        t = sol.t
        x = sol.y
        v = sol.v

        gen_names = list(ps.gen['GEN'].par['name'])

        # Voltage step
        vset = ps.gen['GEN'].v_setp(x, v)
        if t > 1:
            for g in gen_names:
                vset[gen_names.index(g)] = 1.1

        V_B6 = np.abs(v[iB6])
        Q = ps.gen['GEN'].q_e(x, v) * ps.sys_data['s_n']

        try:
            Efd = ps.avr['SEXS'].output(x, v)
        except:
            Efd = ps.avr['SEXS'].u(x, v)

        # VSC Q
        try:
            vsc = ps.vsc['VSC_SI']
            Q_vsc = vsc.q_e(x, v) * vsc.par['S_n']
        except:
            Q_vsc = np.array([0.0])

        res['t'].append(t)
        res['V'].append(V_B6)
        res['Qg_all'].append(Q.copy())
        res['Efd_all'].append(Efd.copy())
        res['Q_vsc'].append(Q_vsc.copy())
        res['gen_names'] = gen_names

    return res


# --------------------------------------------------
# MAIN
# --------------------------------------------------
cases = [0, 1, 2, 4]
results = {}

for n in cases:

    importlib.reload(model_data)
    base_model = model_data.load()

    SCR = compute_scr(base_model, n)
    print(f"n={n} → SCR ≈ {SCR:.2f}")

    model_case = apply_generator_case(base_model, n)
    res = run_simulation(model_case)

    results[n] = {'SCR': SCR, **res}


# --------------------------------------------------
# PLOTS
# --------------------------------------------------

# Voltage
plt.figure()
for n in results:
    plt.plot(results[n]['t'], results[n]['V'],
             label=f"SCR={results[n]['SCR']:.1f}")
plt.title("Voltage at B6")
plt.xlabel("Time [s]")
plt.ylabel("pu")
plt.legend()
plt.grid()

# Average Reactive Power (MAIN)
plt.figure()
for n in results:
    Q_avg = [np.mean(q) for q in results[n]['Qg_all']]
    plt.plot(results[n]['t'], Q_avg,
             label=f"SCR={results[n]['SCR']:.1f}")
   # Average Reactive Power excluding G1
#for n in results:
   # local_idx = [i for i, name in enumerate(results[n]['gen_names']) if name != 'G1']
    #Q_avg = [np.mean(q[local_idx]) for q in results[n]['Qg_all']]
    #plt.plot(results[n]['t'], Q_avg,
         #    label=f"SCR={results[n]['SCR']:.1f}")
plt.title("Average Reactive Power per Generator")
plt.xlabel("Time [s]")
plt.ylabel("MVAr")
plt.legend()
plt.grid()

# Average Field Voltage (MAIN)
plt.figure()
for n in results:
    E_avg = [np.mean(e) for e in results[n]['Efd_all']]
    plt.plot(results[n]['t'], E_avg,
             label=f"SCR={results[n]['SCR']:.1f}")
    #Average Field Voltage excluding G1
#for n in results:
   # local_idx = [i for i, name in enumerate(results[n]['gen_names']) if name != 'G1']
   # E_avg = [np.mean(e[local_idx]) for e in results[n]['Efd_all']]
   # plt.plot(results[n]['t'], E_avg,
             #label=f"SCR={results[n]['SCR']:.1f}")
plt.title("Average Field Voltage per Generator")
plt.xlabel("Time [s]")
plt.ylabel("pu")
plt.legend()
plt.grid()

# VSC Reactive Power
plt.figure()
for n in results:
    Qv = np.array(results[n]['Q_vsc']).squeeze()
    plt.plot(results[n]['t'], Qv,
             label=f"SCR={results[n]['SCR']:.1f}")
plt.title("VSC Reactive Power")
plt.xlabel("Time [s]")
plt.ylabel("MVAr")
plt.legend()
plt.grid()

plt.show()


# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
import importlib
import copy
from collections import defaultdict

import tops.dynamic as dps
import tops.solvers as dps_sol

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_PATH = r"D:\Masters REM+\Master Thesis\paper\TOPS-main\TOPS-main"

sys.path.insert(0, BASE_PATH + r"\examples\Base work")
sys.path.append(BASE_PATH + r"\examples\user_models")

import user_lib
import generator_network as model_data


# --------------------------------------------------
# ONLY STRONG GRID
# --------------------------------------------------
ACTIVE_GENS = ['G1']


# --------------------------------------------------
# EXPORT / IMPORT CASES
# --------------------------------------------------
POWER_CASES = {
    'export': 600,
    'balance': 300,
    'import': 100
}


# --------------------------------------------------
# APPLY GENERATORS
# --------------------------------------------------
def apply_gen_case(model, active_gens):
    model = copy.deepcopy(model)

    header = model['generators']['GEN'][0]
    model['generators']['GEN'] = [header] + [
        g for g in model['generators']['GEN'][1:] if g[0] in active_gens
    ]

    for key in ['gov', 'avr', 'pss']:
        subkey = list(model[key].keys())[0]
        header = model[key][subkey][0]
        model[key][subkey] = [header] + [
            g for g in model[key][subkey][1:] if g[1] in active_gens
        ]

    return model


# --------------------------------------------------
# DISTRIBUTE POWER
# --------------------------------------------------
def distribute_power(model, total_power, active_gens):
    model = copy.deepcopy(model)

    local_gens = [g for g in active_gens if g != 'G1']
    n = len(local_gens)

    if n == 0:
        return model

    share = total_power / n

    header = model['generators']['GEN'][0]
    P_idx = header.index('P')

    for g in model['generators']['GEN'][1:]:
        if g[0] in local_gens:
            g[P_idx] = share

    return model


# --------------------------------------------------
# SAFE VSC READ
# --------------------------------------------------
def read_vsc(ps, x, v):
    if 'VSC_SI' not in ps.vsc:
        return 0.0, 0.0

    vsc = ps.vsc['VSC_SI']
    s_base = ps.sys_data['s_n']

    P = vsc.p_e(x, v) * s_base
    Q = vsc.q_e(x, v) * s_base

    return float(np.asarray(P).item()), float(np.asarray(Q).item())


# --------------------------------------------------
# SIMULATION
# --------------------------------------------------
def run_simulation(model):
    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_lib)

    ps.power_flow()
    ps.init_dyn_sim()

    sol = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0,
        ps.x_0.copy(),
        10,
        max_step=2e-3
    )

    idx = {'B1':0,'B2':1,'B5':2,'B6':3,'B7':4,'B8':5,'B9':6,'B10':7}
    iB6 = idx['B6']

    res = defaultdict(list)
    t = 0.0

    while t < 10:
        sol.step()

        t = sol.t
        x = sol.y
        v = sol.v

        gen_names = list(ps.gen['GEN'].par['name'])

        # Voltage step
        vset = ps.gen['GEN'].v_setp(x, v)
        if t > 1 and not hasattr(ps, "step_done"):
            for i in range(len(gen_names)):
                vset[i] = 1.1
            ps.step_done = True

        # Measurements
        V = np.abs(v[iB6])

        Pg = ps.gen['GEN'].p_e(x, v) * ps.sys_data['s_n']
        Qg = ps.gen['GEN'].q_e(x, v) * ps.sys_data['s_n']

        Pg1 = Pg[gen_names.index('G1')]
        Qg1 = Qg[gen_names.index('G1')]

        Pg3 = np.nan
        Qg3 = np.nan
        if 'G3' in gen_names:
            Pg3 = Pg[gen_names.index('G3')]
            Qg3 = Qg[gen_names.index('G3')]

        Q_total = np.sum(Qg)
        Q_avg = np.mean(Qg)

        Pvsc, Qvsc = read_vsc(ps, x, v)

        res['t'].append(t)
        res['V'].append(V)
        res['Qg1'].append(Qg1)
        res['Qg3'].append(Qg3)
        res['Q_total'].append(Q_total)
        res['Q_avg'].append(Q_avg)
        res['Qvsc'].append(Qvsc)
        res['Pg1'].append(Pg1)

    return res


# --------------------------------------------------
# MAIN
# --------------------------------------------------
results = {}

for label, P in POWER_CASES.items():
    importlib.reload(model_data)
    base_model = model_data.load()

    model_case = apply_gen_case(base_model, ACTIVE_GENS)
    model_case = distribute_power(model_case, P, ACTIVE_GENS)

    print(f"\nRunning {label} case → {P} MW")

    results[label] = run_simulation(model_case)


# --------------------------------------------------
# PLOTS
# --------------------------------------------------
plt.figure()
for k in results:
    plt.plot(results[k]['t'], results[k]['V'], label=k)
plt.title("Voltage at B6")
plt.xlabel("Time [s]")
plt.ylabel("pu")
plt.legend()
plt.grid()

plt.figure()
for k in results:
    plt.plot(results[k]['t'], results[k]['Qg1'], label=k)
plt.title("Reactive Power of G1 (Slack)")
plt.xlabel("Time [s]")
plt.ylabel("MVAr")
plt.legend()
plt.grid()

plt.figure()
for k in results:
    plt.plot(results[k]['t'], results[k]['Q_avg'], label=k)
plt.title("Average Reactive Power of Online Generators")
plt.xlabel("Time [s]")
plt.ylabel("MVAr")
plt.legend()
plt.grid()

plt.figure()
for k in results:
    plt.plot(results[k]['t'], results[k]['Qvsc'], label=k)
plt.title("VSC Reactive Power")
plt.xlabel("Time [s]")
plt.ylabel("MVAr")
plt.legend()
plt.grid()

plt.figure()
for k in results:
    plt.plot(results[k]['t'], results[k]['Pg1'], label=k)
plt.title("Slack Generator Active Power")
plt.xlabel("Time [s]")
plt.ylabel("MW")
plt.legend()
plt.grid()

plt.show()


# ================================
# CASE: G1 + G3
# ================================

import sys
import numpy as np
import matplotlib.pyplot as plt
import importlib
import copy
from collections import defaultdict

import tops.dynamic as dps
import tops.solvers as dps_sol

BASE_PATH = r"D:\Masters REM+\Master Thesis\paper\TOPS-main\TOPS-main"

sys.path.insert(0, BASE_PATH + r"\examples\Base work")
sys.path.append(BASE_PATH + r"\examples\user_models")

import user_lib
import generator_network as model_data


ACTIVE_GENS = ['G1', 'G3']
TITLE = "Generator Case: G1 + G3"

POWER_CASES = {
    'export': 600,
    'balance': 300,
    'import': 100
}


# ------------------------------
# Apply generators
# ------------------------------
def apply_gen_case(model):
    model = copy.deepcopy(model)

    def filter_block(block, idx):
        header = block[0]
        return [header] + [row for row in block[1:] if row[idx] in ACTIVE_GENS]

    model['generators']['GEN'] = filter_block(model['generators']['GEN'], 0)
    model['gov']['HYGOV'] = filter_block(model['gov']['HYGOV'], 1)
    model['avr']['SEXS'] = filter_block(model['avr']['SEXS'], 1)
    model['pss']['STAB1'] = filter_block(model['pss']['STAB1'], 1)

    return model


# ------------------------------
# Distribute power
# ------------------------------
def apply_power(model, total_P):
    model = copy.deepcopy(model)

    local_gens = [g for g in ACTIVE_GENS if g != 'G1']
    share = total_P / len(local_gens)

    header = model['generators']['GEN'][0]
    P_idx = header.index('P')

    for row in model['generators']['GEN'][1:]:
        if row[0] in local_gens:
            row[P_idx] = share

    return model


# ------------------------------
# Simulation
# ------------------------------
def run(model):
    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_lib)
    ps.power_flow()
    ps.init_dyn_sim()

    sol = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0,
        ps.x_0.copy(),
        10,
        max_step=2e-3
    )

    idx = {'B6': 3}
    iB6 = idx['B6']

    res = defaultdict(list)

    while sol.t < 10:
        sol.step()

        x, v = sol.y, sol.v
        gen_names = list(ps.gen['GEN'].par['name'])

        if sol.t > 1 and not hasattr(ps, "step_done"):
            vset = ps.gen['GEN'].v_setp(x, v)
            for i in range(len(vset)):
                vset[i] = 1.1
            ps.step_done = True

        V = np.abs(v[iB6])
        Q = ps.gen['GEN'].q_e(x, v) * ps.sys_data['s_n']

        res['t'].append(sol.t)
        res['V'].append(V)
        res['Qavg'].append(np.mean(Q))

        if 'G3' in gen_names:
            res['Qg3'].append(Q[gen_names.index('G3')])
        else:
            res['Qg3'].append(0)

        # VSC
        try:
            Qvsc = ps.vsc['VSC_SI'].q_e(x, v) * ps.sys_data['s_n']
            res['Qvsc'].append(float(np.asarray(Qvsc).flatten()[0]))
        except:
            res['Qvsc'].append(0)

    return res


# ------------------------------
# MAIN
# ------------------------------
results = {}

for k, P in POWER_CASES.items():
    importlib.reload(model_data)
    m = model_data.load()

    m = apply_gen_case(m)
    m = apply_power(m, P)

    print(f"{k} → {P} MW")

    results[k] = run(m)


# ------------------------------
# PLOTS
# ------------------------------
for key, ylabel, title in [
    ('V', 'pu', 'Voltage at B6'),
    ('Qavg', 'MVAr', 'Avg Reactive Power'),
    ('Qg3', 'MVAr', 'Reactive Power of G3'),
    ('Qvsc', 'MVAr', 'VSC Reactive Power')
]:
    plt.figure()
    for k in results:
        plt.plot(results[k]['t'], results[k][key], label=k)
    plt.title(f"{TITLE} - {title}")
    plt.xlabel("Time [s]")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()

plt.show()

# ================================
# CASE: G1 + G3 + G4
# ================================

import sys
import numpy as np
import matplotlib.pyplot as plt
import importlib
import copy
from collections import defaultdict

import tops.dynamic as dps
import tops.solvers as dps_sol

BASE_PATH = r"D:\Masters REM+\Master Thesis\paper\TOPS-main\TOPS-main"

sys.path.insert(0, BASE_PATH + r"\examples\Base work")
sys.path.append(BASE_PATH + r"\examples\user_models")

import user_lib
import generator_network as model_data
# SAME imports as before...

ACTIVE_GENS = ['G1', 'G3', 'G4']
TITLE = "Generator Case: G1 + G3 + G4"

POWER_CASES = {
    'export': 600,
    'balance': 300,
    'import': 100
}


def apply_gen_case(model):
    model = copy.deepcopy(model)

    def filter_block(block, idx):
        header = block[0]
        return [header] + [row for row in block[1:] if row[idx] in ACTIVE_GENS]

    model['generators']['GEN'] = filter_block(model['generators']['GEN'], 0)
    model['gov']['HYGOV'] = filter_block(model['gov']['HYGOV'], 1)
    model['avr']['SEXS'] = filter_block(model['avr']['SEXS'], 1)
    model['pss']['STAB1'] = filter_block(model['pss']['STAB1'], 1)

    return model


def apply_power(model, total_P):
    model = copy.deepcopy(model)

    local_gens = [g for g in ACTIVE_GENS if g != 'G1']
    share = total_P / len(local_gens)   # <-- important

    header = model['generators']['GEN'][0]
    P_idx = header.index('P')

    for row in model['generators']['GEN'][1:]:
        if row[0] in local_gens:
            row[P_idx] = share

    return model


def run(model):
    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_lib)
    ps.power_flow()
    ps.init_dyn_sim()

    sol = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0,
        ps.x_0.copy(),
        10,
        max_step=2e-3
    )

    idx = {'B6': 3}
    iB6 = idx['B6']

    res = defaultdict(list)

    while sol.t < 10:
        sol.step()

        x, v = sol.y, sol.v
        gen_names = list(ps.gen['GEN'].par['name'])

        if sol.t > 1 and not hasattr(ps, "step_done"):
            vset = ps.gen['GEN'].v_setp(x, v)
            for i in range(len(vset)):
                vset[i] = 1.1
            ps.step_done = True

        V = np.abs(v[iB6])
        Q = ps.gen['GEN'].q_e(x, v) * ps.sys_data['s_n']

        res['t'].append(sol.t)
        res['V'].append(V)
        res['Qavg'].append(np.mean(Q))

        # Track G3
        res['Qg3'].append(Q[gen_names.index('G3')])

        try:
            Qvsc = ps.vsc['VSC_SI'].q_e(x, v) * ps.sys_data['s_n']
            res['Qvsc'].append(float(Qvsc))
        except:
            res['Qvsc'].append(0)

    return res


# MAIN
results = {}

for k, P in POWER_CASES.items():
    importlib.reload(model_data)
    m = model_data.load()

    m = apply_gen_case(m)
    m = apply_power(m, P)

    print(f"{k} → {P} MW")

    results[k] = run(m)


# PLOTS
for key, ylabel, title in [
    ('V', 'pu', 'Voltage at B6'),
    ('Qavg', 'MVAr', 'Avg Reactive Power'),
    ('Qg3', 'MVAr', 'Reactive Power of G3'),
    ('Qvsc', 'MVAr', 'VSC Reactive Power')
]:
    plt.figure()
    for k in results:
        plt.plot(results[k]['t'], results[k][key], label=k)
    plt.title(f"{TITLE} - {title}")
    plt.xlabel("Time [s]")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()

plt.show()


# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
import importlib
import copy
from collections import defaultdict

import tops.dynamic as dps
import tops.solvers as dps_sol

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_PATH = r"D:\Masters REM+\Master Thesis\paper\TOPS-main\TOPS-main"

sys.path.insert(0, BASE_PATH + r"\examples\Base work")
sys.path.append(BASE_PATH + r"\examples\user_models")

import user_lib
import generator_network as model_data


# --------------------------------------------------
# FIXED STRONG GRID (CASE 4)
# --------------------------------------------------
ACTIVE_GENS = ['G1', 'G3', 'G4', 'G5', 'G6']


# --------------------------------------------------
# EXPORT / IMPORT CASES
# --------------------------------------------------
POWER_CASES = {
    'export': 600,
    'balance': 300,
    'import': 100
}


# --------------------------------------------------
# APPLY GENERATORS
# --------------------------------------------------
def apply_gen_case(model, active_gens):
    model = copy.deepcopy(model)

    header = model['generators']['GEN'][0]
    model['generators']['GEN'] = [header] + [
        g for g in model['generators']['GEN'][1:] if g[0] in active_gens
    ]

    for key in ['gov', 'avr', 'pss']:
        subkey = list(model[key].keys())[0]
        header = model[key][subkey][0]
        model[key][subkey] = [header] + [
            g for g in model[key][subkey][1:] if g[1] in active_gens
        ]

    return model


# --------------------------------------------------
# DISTRIBUTE POWER
# --------------------------------------------------
def distribute_power(model, total_power, active_gens):
    model = copy.deepcopy(model)

    local_gens = [g for g in active_gens if g != 'G1']
    n = len(local_gens)

    share = total_power / n

    header = model['generators']['GEN'][0]
    P_idx = header.index('P')

    for g in model['generators']['GEN'][1:]:
        if g[0] in local_gens:
            g[P_idx] = share

    return model


# --------------------------------------------------
# SAFE VSC READ
# --------------------------------------------------
def read_vsc(ps, x, v):
    if 'VSC_SI' not in ps.vsc:
        return 0.0, 0.0

    vsc = ps.vsc['VSC_SI']
    s_base = ps.sys_data['s_n']

    P = vsc.p_e(x, v) * s_base
    Q = vsc.q_e(x, v) * s_base

    return float(np.asarray(P).item()), float(np.asarray(Q).item())


# --------------------------------------------------
# SIMULATION
# --------------------------------------------------
def run_simulation(model):

    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_lib)

    ps.power_flow()
    ps.init_dyn_sim()

    sol = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0,
        ps.x_0.copy(),
        10,
        max_step=2e-3
    )

    idx = {'B1':0,'B2':1,'B5':2,'B6':3,'B7':4,'B8':5,'B9':6,'B10':7}
    iB6 = idx['B6']

    res = defaultdict(list)
    t = 0.0

    while t < 10:

        sol.step()

        t = sol.t
        x = sol.y
        v = sol.v

        gen_names = list(ps.gen['GEN'].par['name'])

        # ---------------- Voltage step ----------------
        vset = ps.gen['GEN'].v_setp(x, v)
        if t > 1 and not hasattr(ps, "step_done"):
            for i in range(len(gen_names)):
                vset[i] = 1.1
            ps.step_done = True

        # ---------------- Measurements ----------------
        V = np.abs(v[iB6])

        Pg = ps.gen['GEN'].p_e(x, v) * ps.sys_data['s_n']
        Qg = ps.gen['GEN'].q_e(x, v) * ps.sys_data['s_n']

        # Individual generators
        Pg1 = Pg[gen_names.index('G1')]
        Qg1 = Qg[gen_names.index('G1')]

        Pg3 = Pg[gen_names.index('G3')]
        Qg3 = Qg[gen_names.index('G3')]

        # Totals / averages
        Q_total = np.sum(Qg)
        Q_avg = np.mean(Qg)

        # VSC
        Pvsc, Qvsc = read_vsc(ps, x, v)

        # Store
        res['t'].append(t)
        res['V'].append(V)

        res['Qg1'].append(Qg1)
        res['Qg3'].append(Qg3)

        res['Q_total'].append(Q_total)
        res['Q_avg'].append(Q_avg)

        res['Qvsc'].append(Qvsc)
        res['Pg1'].append(Pg1)   # slack → power transfer

    return res


# --------------------------------------------------
# MAIN
# --------------------------------------------------
results = {}

for label, P in POWER_CASES.items():

    importlib.reload(model_data)
    base_model = model_data.load()

    model_case = apply_gen_case(base_model, ACTIVE_GENS)
    model_case = distribute_power(model_case, P, ACTIVE_GENS)

    print(f"\nRunning {label} case → {P} MW")

    results[label] = run_simulation(model_case)


# --------------------------------------------------
# PLOTS
# --------------------------------------------------

# Voltage
plt.figure()
for k in results:
    plt.plot(results[k]['t'], results[k]['V'], label=k)
plt.title("Voltage at B6")
plt.legend()
plt.grid()

# Generator Reactive Power (G3)
plt.figure()
for k in results:
    plt.plot(results[k]['t'], results[k]['Qg3'], label=k)
plt.title("Reactive Power of G3")
plt.legend()
plt.grid()

# Generator Reactive Power (G1)
plt.figure()
for k in results:
    plt.plot(results[k]['t'], results[k]['Qg1'], label=k)
plt.title("Reactive Power of G1 (Slack)")
plt.legend()
plt.grid()

# Average Q (IMPORTANT)
plt.figure()
for k in results:
    plt.plot(results[k]['t'], results[k]['Q_avg'], label=k)
plt.title("Average Reactive Power per Generator")
plt.legend()
plt.grid()

# VSC Reactive Power
plt.figure()
for k in results:
    plt.plot(results[k]['t'], results[k]['Qvsc'], label=k)
plt.title("VSC Reactive Power")
plt.legend()
plt.grid()

# Power transfer (G1)
plt.figure()
for k in results:
    plt.plot(results[k]['t'], results[k]['Pg1'], label=k)
plt.title("Slack Generator Power (Power Transfer)")
plt.legend()
plt.grid()

plt.show()



# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
import importlib
import copy

import tops.dynamic as dps
import tops.solvers as dps_sol

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_PATH = r"D:\Masters REM+\Master Thesis\paper\TOPS-main\TOPS-main"
sys.path.insert(0, BASE_PATH + r"\examples\Base work")
sys.path.append(BASE_PATH + r"\examples\user_models")

import user_lib
import generator_network as model_data


# --------------------------------------------------
# STUDY SETTINGS
# --------------------------------------------------
LOCAL_UNIT_RATING_MW = 360.0
LOADING_PU = 0.85
BASE_LOCAL_UNIT_MW = LOCAL_UNIT_RATING_MW * LOADING_PU   # 306 MW per active local generator

FAULT_START = 1.0
SIM_END = 8.0
MAX_STEP = 5e-3
FAULT_Y = 1e6

BUS_IDX = {
    'B1': 0,
    'B2': 1,
    'B5': 2,
    'B6': 3,
    'B7': 4,
    'B8': 5,
    'B9': 6,
    'B10': 7
}

FAULT_BUS = 'B6'
FAULT_BUS_INDEX = BUS_IDX[FAULT_BUS]

GEN_CASES = {
    '1-gen': ['G1', 'G3'],
    '2-gen': ['G1', 'G3', 'G4'],
    '4-gen': ['G1', 'G3', 'G4', 'G5', 'G6'],
}

COARSE_DURATIONS = np.concatenate([
    np.arange(0.05, 1.05, 0.05),
    np.arange(1.10, 2.10, 0.10),
    np.arange(2.20, 3.20, 0.20),
])

REFINE_TOL = 1e-3


# --------------------------------------------------
# CASE PREPARATION
# --------------------------------------------------
def apply_case(model, active_gens):
    model = copy.deepcopy(model)
    keep = set(active_gens)

    # generators
    header = model['generators']['GEN'][0]
    model['generators']['GEN'] = [header] + [
        row for row in model['generators']['GEN'][1:] if row[0] in keep
    ]

    # governors
    if 'gov' in model and 'HYGOV' in model['gov']:
        header = model['gov']['HYGOV'][0]
        model['gov']['HYGOV'] = [header] + [
            row for row in model['gov']['HYGOV'][1:] if row[1] in keep
        ]

    # AVR
    if 'avr' in model and 'SEXS' in model['avr']:
        header = model['avr']['SEXS'][0]
        model['avr']['SEXS'] = [header] + [
            row for row in model['avr']['SEXS'][1:] if row[1] in keep
        ]

    # PSS
    if 'pss' in model and 'STAB1' in model['pss']:
        header = model['pss']['STAB1'][0]
        model['pss']['STAB1'] = [header] + [
            row for row in model['pss']['STAB1'][1:] if row[1] in keep
        ]

    return model


def set_local_generation(model, active_gens, p_each=BASE_LOCAL_UNIT_MW):
    """
    G1 stays at its original value.
    Every active local generator gets the same MW loading.
    """
    model = copy.deepcopy(model)
    header = model['generators']['GEN'][0]
    P_idx = header.index('P')

    for row in model['generators']['GEN'][1:]:
        gname = row[0]
        if gname != 'G1' and gname in active_gens:
            row[P_idx] = p_each

    return model


def scale_t4(model, n_local):
    """
    Transformer scaling rule:
      n=1 -> T4 S_n = 360
      n=2 -> T4 S_n = 720
      n=4 -> T4 S_n = 1440
    """
    model = copy.deepcopy(model)

    header = model['transformers'][0]
    S_idx = header.index('S_n')

    for row in model['transformers'][1:]:
        if row[0] == 'T4':
            if n_local == 1:
                row[S_idx] = 360
            elif n_local == 2:
                row[S_idx] = 720
            elif n_local == 4:
                row[S_idx] = 1440

    return model


# --------------------------------------------------
# SAFE VSC READ
# --------------------------------------------------
def read_vsc_q(ps, x, v):
    try:
        if hasattr(ps, 'vsc') and 'VSC_SI' in ps.vsc:
            return float(np.asarray(ps.vsc['VSC_SI'].q_e(x, v)).item()) * ps.sys_data['s_n']
    except:
        pass
    return 0.0


# --------------------------------------------------
# ONE DYNAMIC RUN
# --------------------------------------------------
def run_sim(active_gens, fault_duration, record=False):
    """
    fault_duration = duration the fault remains applied after FAULT_START
    actual clearing instant = FAULT_START + fault_duration
    """
    importlib.reload(model_data)
    model = model_data.load()

    n_local = len([g for g in active_gens if g != 'G1'])

    model = apply_case(model, active_gens)
    model = set_local_generation(model, active_gens, p_each=BASE_LOCAL_UNIT_MW)
    model = scale_t4(model, n_local)

    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_lib)
    ps.power_flow()
    ps.init_dyn_sim()

    sol = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0,
        ps.x_0.copy(),
        SIM_END,
        max_step=MAX_STEP
    )

    gen_names = list(ps.gen['GEN'].par['name'])
    iG1 = gen_names.index('G1')
    local_names = [g for g in active_gens if g != 'G1']
    local_idx = [gen_names.index(g) for g in local_names]

    res = {
        't': [],
        'delta_group': [],
        'delta_each': {g: [] for g in local_names},
        'Vb6': [],
        'Qg1': [],
        'Qlocal': {g: [] for g in local_names},
        'Qvsc': [],
        'speed_group': [],
    }

    stable = True

    while sol.t < SIM_END:
        t_now = sol.t

        if FAULT_START <= t_now < FAULT_START + fault_duration:
            ps.y_bus_red_mod[(FAULT_BUS_INDEX, FAULT_BUS_INDEX)] = FAULT_Y
        else:
            ps.y_bus_red_mod[(FAULT_BUS_INDEX, FAULT_BUS_INDEX)] = 0

        sol.step()

        t = sol.t
        x = sol.y
        v = sol.v

        angles = np.asarray(ps.gen['GEN'].angle(x, v), dtype=float)
        speeds = np.asarray(ps.gen['GEN'].speed(x, v), dtype=float)
        q_gen = np.asarray(ps.gen['GEN'].q_e(x, v), dtype=float) * ps.sys_data['s_n']

        delta_each = {}
        for g, idx in zip(local_names, local_idx):
            delta_each[g] = float(angles[idx] - angles[iG1])

        delta_group = float(np.mean([delta_each[g] for g in local_names]))
        speed_group = float(np.mean([speeds[idx] for idx in local_idx])) if local_idx else 1.0

        if any(abs(delta_each[g]) > np.pi for g in local_names):
            stable = False
            if record:
                res['t'].append(t)
                res['delta_group'].append(delta_group)
                for g in local_names:
                    res['delta_each'][g].append(delta_each[g])
                res['Vb6'].append(float(np.abs(v[FAULT_BUS_INDEX])))
                res['Qg1'].append(float(q_gen[iG1]))
                for g, idx in zip(local_names, local_idx):
                    res['Qlocal'][g].append(float(q_gen[idx]))
                res['Qvsc'].append(read_vsc_q(ps, x, v))
                res['speed_group'].append(speed_group)
            break

        if record:
            res['t'].append(t)
            res['delta_group'].append(delta_group)
            for g in local_names:
                res['delta_each'][g].append(delta_each[g])
            res['Vb6'].append(float(np.abs(v[FAULT_BUS_INDEX])))
            res['Qg1'].append(float(q_gen[iG1]))
            for g, idx in zip(local_names, local_idx):
                res['Qlocal'][g].append(float(q_gen[idx]))
            res['Qvsc'].append(read_vsc_q(ps, x, v))
            res['speed_group'].append(speed_group)

    return stable, res


# --------------------------------------------------
# CCT SEARCH
# --------------------------------------------------
def find_cct_for_case(label, active_gens):
    print(f"\nCase: {label} | Active generators: {active_gens}")

    last_stable = None
    first_unstable = None

    for dur in COARSE_DURATIONS:
        stable, _ = run_sim(active_gens, dur, record=False)
        print(f"  fault duration = {dur:.3f} s -> {'STABLE' if stable else 'UNSTABLE'}")

        if stable:
            last_stable = dur
        else:
            first_unstable = dur
            break

    if last_stable is None:
        return {
            'status': 'no stable case found',
            'cct_duration': None,
            'clear_time': None,
            'stable_res': None,
            'unstable_res': None,
        }

    if first_unstable is None:
        return {
            'status': f'no unstable case found up to {last_stable:.3f} s',
            'cct_duration': last_stable,
            'clear_time': FAULT_START + last_stable,
            'stable_res': run_sim(active_gens, last_stable, record=True)[1],
            'unstable_res': None,
        }

    lo = last_stable
    hi = first_unstable

    while (hi - lo) > REFINE_TOL:
        mid = 0.5 * (lo + hi)
        stable, _ = run_sim(active_gens, mid, record=False)
        print(f"    refine {mid:.4f} s -> {'STABLE' if stable else 'UNSTABLE'}")

        if stable:
            lo = mid
        else:
            hi = mid

    cct_duration = lo
    stable_res = run_sim(active_gens, max(0.01, cct_duration - 0.005), record=True)[1]
    unstable_res = run_sim(active_gens, hi, record=True)[1]

    return {
        'status': 'ok',
        'cct_duration': cct_duration,
        'clear_time': FAULT_START + cct_duration,
        'stable_res': stable_res,
        'unstable_res': unstable_res,
    }


# --------------------------------------------------
# PLOTTING
# --------------------------------------------------
def plot_case_result(label, case_result, active_gens):
    stable_res = case_result['stable_res']
    unstable_res = case_result['unstable_res']

    if stable_res is None and unstable_res is None:
        return

    local_names = [g for g in active_gens if g != 'G1']

    # 1) group angle
    plt.figure()
    if stable_res is not None:
        plt.plot(stable_res['t'], stable_res['delta_group'], linewidth=2, label='Stable case')
    if unstable_res is not None:
        plt.plot(unstable_res['t'], unstable_res['delta_group'], '--', linewidth=2, label='Unstable case')
    plt.axhline(np.pi, linestyle=':', color='k', label='±π limit')
    plt.axhline(-np.pi, linestyle=':', color='k')
    plt.title(f"CCT Analysis ({label}): Average Local Rotor Angle Relative to G1")
    plt.xlabel("Time [s]")
    plt.ylabel("Rotor angle [rad]")
    plt.grid()
    plt.legend()

    # 2) each local machine relative to G1
    plt.figure()
    if stable_res is not None:
        for g in local_names:
            plt.plot(stable_res['t'], stable_res['delta_each'][g], linewidth=2, label=f'{g} stable')
    if unstable_res is not None:
        for g in local_names:
            plt.plot(unstable_res['t'], unstable_res['delta_each'][g], '--', linewidth=2, label=f'{g} unstable')
    plt.axhline(np.pi, linestyle=':', color='k', label='±π limit')
    plt.axhline(-np.pi, linestyle=':', color='k')
    plt.title(f"CCT Analysis ({label}): Individual Local Generator Angles Relative to G1")
    plt.xlabel("Time [s]")
    plt.ylabel("Rotor angle [rad]")
    plt.grid()
    plt.legend()

    # 3) B6 voltage
    plt.figure()
    if stable_res is not None:
        plt.plot(stable_res['t'], stable_res['Vb6'], linewidth=2, label='Stable case')
    if unstable_res is not None:
        plt.plot(unstable_res['t'], unstable_res['Vb6'], '--', linewidth=2, label='Unstable case')
    plt.title(f"CCT Analysis ({label}): Voltage Response at Bus {FAULT_BUS}")
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage magnitude [pu]")
    plt.grid()
    plt.legend()

    # 4) VSC Q
    plt.figure()
    if stable_res is not None:
        plt.plot(stable_res['t'], stable_res['Qvsc'], linewidth=2, label='Stable case')
    if unstable_res is not None:
        plt.plot(unstable_res['t'], unstable_res['Qvsc'], '--', linewidth=2, label='Unstable case')
    plt.title(f"CCT Analysis ({label}): VSC Reactive Power Response")
    plt.xlabel("Time [s]")
    plt.ylabel("Reactive power [MVAr]")
    plt.grid()
    plt.legend()

    # 5) local group speed
    plt.figure()
    if stable_res is not None:
        plt.plot(stable_res['t'], stable_res['speed_group'], linewidth=2, label='Stable case')
    if unstable_res is not None:
        plt.plot(unstable_res['t'], unstable_res['speed_group'], '--', linewidth=2, label='Unstable case')
    plt.title(f"CCT Analysis ({label}): Average Local Generator Speed")
    plt.xlabel("Time [s]")
    plt.ylabel("Speed [pu]")
    plt.grid()
    plt.legend()


# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    print("\n========== CCT SEARCH ==========")

    all_results = {}

    for label, gens in GEN_CASES.items():
        all_results[label] = find_cct_for_case(label, gens)

    print("\n========== FINAL RESULTS ==========")
    for label, out in all_results.items():
        print(f"{label}: {out['status']}")
        print(f"  Critical fault duration = {out['cct_duration']}")
        print(f"  Critical clearing instant = {out['clear_time']}")

    for label, gens in GEN_CASES.items():
        plot_case_result(label, all_results[label], gens)

    plt.show()
    
    
    
    chechk the whole thesis step by step....this is my thesis work...