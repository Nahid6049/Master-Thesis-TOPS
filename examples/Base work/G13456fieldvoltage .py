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
# SETTINGS
# --------------------------------------------------
LOCAL_UNIT_RATING = 360      # MVA
VSC_RATING_MVA = 1000        # MVA


# --------------------------------------------------
# SCR CALCULATION
# --------------------------------------------------
def compute_scr(model, n):

    Z_L56 = Z_L25 = Z_L69 = None
    Z_T1 = Z_T4 = None
    Z_G1 = None
    Z_G3 = None

    S_base = model["base_mva"]   # 1000 MVA

    # Lines
    for line in model["lines"][1:]:
        Z = line[7] + 1j * line[8]

        if line[0] == "L5-6":
            Z_L56 = Z
        elif line[0] == "L2-5":
            Z_L25 = Z
        elif line[0] == "L6-9":
            Z_L69 = Z

    # Transformers
    for tr in model["transformers"][1:]:
        Z = tr[6] + 1j * tr[7]

        if tr[0] == "T1":
            Z_T1 = Z
        elif tr[0] == "T4":
            Z_T4 = Z

    # Generators
    for gen in model["generators"]["GEN"][1:]:
        if gen[0] == "G1":
            Z_G1 = 1j * gen[12]   # Xd''
        elif gen[0] == "G3":
            Z_G3 = 1j * gen[12]   # representative local generator Xd''

    # External grid branch is kept as in the implemented equivalent model
    Z_grid = Z_L25 + Z_L56 + Z_T1 + Z_G1

    if n == 0 or Z_G3 is None:
        Z_th = Z_grid

    else:
        # Local branch before conversion:
        # line + transformer + generator subtransient reactance
        Z_local_old = Z_L69 + Z_T4 + Z_G3

        # Convert local branch from 360 MVA base to 1000 MVA base
        Z_local_new = Z_local_old * (S_base / LOCAL_UNIT_RATING)

        # Equivalent of n local units in parallel
        Z_local_eq = Z_local_new / n

        # Thevenin equivalent at PCC
        Z_th = 1 / ((1 / Z_grid) + (1 / Z_local_eq))

    # Short-circuit power in pu on system base
    S_sc = 1 / abs(Z_th)

    # Converter rating in pu
    S_conv_pu = VSC_RATING_MVA / S_base

    # Short-circuit ratio
    SCR = S_sc / S_conv_pu

    return SCR


# --------------------------------------------------
# APPLY GENERATOR CASE + TRANSFORMER SCALING
def apply_generator_case(model, n):

    model = copy.deepcopy(model)

    # Select generators
    if n == 0:
        keep = {"G1"}
    elif n == 1:
        keep = {"G1", "G3"}
    elif n == 2:
        keep = {"G1", "G3", "G4"}
    elif n == 4:
        keep = {"G1", "G3", "G4", "G5", "G6"}
    else:
        raise ValueError("Unsupported case. Use n = 0, 1, 2, or 4.")

    # Generators
    header = model["generators"]["GEN"][0]
    model["generators"]["GEN"] = [header] + [
        g for g in model["generators"]["GEN"][1:] if g[0] in keep
    ]

    # T4 scaling
    if n > 0:
        for tr in model["transformers"][1:]:
            if tr[0] == "T4":
                tr[3] = LOCAL_UNIT_RATING * n
                print(f"T4 scaled to {tr[3]:.0f} MVA")
    
    # Governor / AVR / PSS
    for key in ["gov", "avr", "pss"]:
        subkey = list(model[key].keys())[0]
        sub_header = model[key][subkey][0]

        model[key][subkey] = [sub_header] + [
            g for g in model[key][subkey][1:] if g[1] in keep
        ]
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

    idx = {
        "B1": 0,
        "B2": 1,
        "B5": 2,
        "B6": 3,
        "B7": 4,
        "B8": 5,
        "B9": 6,
        "B10": 7
    }

    iB6 = idx["B6"]

    res = defaultdict(list)
    t = 0.0
    step_done = False

    while t < 10:

        sol.step()

        t = sol.t
        x = sol.y
        v = sol.v

        gen_names = list(ps.gen["GEN"].par["name"])

        # Voltage reference step applied only once after t > 1 s
        vset = ps.gen["GEN"].v_setp(x, v)

        if (t > 1.0) and (not step_done):
            for i in range(len(gen_names)):
                vset[i] = 1.02
            step_done = True

        V_B6 = np.abs(v[iB6])

        Q = ps.gen["GEN"].q_e(x, v) * ps.sys_data["s_n"]

        try:
            Efd = ps.avr["SEXS"].output(x, v)
        except Exception:
            Efd = ps.avr["SEXS"].u(x, v)

       
        # G3 reactive power and field voltage
        # For SCR = 2.8 case, G3 is not connected.
        # Therefore NaN is stored and no G3 curve appears.
        
        if "G3" in gen_names:
            g3_idx = gen_names.index("G3")
            Q_G3 = float(Q[g3_idx])
            Efd_G3 = float(Efd[g3_idx])
        else:
            Q_G3 = np.nan
            Efd_G3 = np.nan

        # VSC reactive power
        try:
            vsc = ps.vsc["VSC_SI"]
            Q_vsc = vsc.q_e(x, v) * vsc.par["S_n"]
        except Exception:
            Q_vsc = np.array([0.0])

        res["t"].append(t)
        res["V"].append(V_B6)

        res["Qg_all"].append(np.array(Q, dtype=float).copy())
        res["Efd_all"].append(np.array(Efd, dtype=float).copy())

        res["Q_G3"].append(Q_G3)
        res["Efd_G3"].append(Efd_G3)

        res["Q_vsc"].append(np.array(Q_vsc, dtype=float).copy())
        res["gen_names"] = gen_names

    return res


# --------------------------------------------------
# MAIN
# --------------------------------------------------
# --------------------------------------------------
# MAIN
# --------------------------------------------------
cases = [0, 1, 2, 4]
results = {}

for n in cases:

    importlib.reload(model_data)
    base_model = model_data.load()

    # Apply generator selection + T4 scaling first
    model_case = apply_generator_case(base_model, n)

    # Then compute SCR using the final case model
    SCR = compute_scr(model_case, n)

    print("\n----------------------------------")
    print(f"Case n = {n}")
    print(f"SCR ≈ {SCR:.2f}")

    res = run_simulation(model_case)

    results[n] = {
        "SCR": SCR,
        **res
    }


# --------------------------------------------------
# PLOTS
# --------------------------------------------------

# 1. Voltage at B6
plt.figure()
for n in results:
    plt.plot(
        results[n]["t"],
        results[n]["V"],
        label=f"SCR={results[n]['SCR']:.1f}"
    )

plt.title("Voltage at B6")
plt.xlabel("Time [s]")
plt.ylabel("pu")
plt.legend()


# 2. Reactive Power of G3
plt.figure()
for n in results:

    Q_g3 = np.array(results[n]["Q_G3"], dtype=float)

    # Skip SCR=2.8 case because G3 is not connected
    if np.all(np.isnan(Q_g3)):
        continue

    plt.plot(
        results[n]["t"],
        Q_g3,
        label=f"SCR={results[n]['SCR']:.1f}"
    )

plt.title("Reactive Power of G3")
plt.xlabel("Time [s]")
plt.ylabel("MVAr")
plt.legend()


# 3. Field Voltage of G3
plt.figure()
for n in results:

    Efd_g3 = np.array(results[n]["Efd_G3"], dtype=float)

    # Skip SCR=2.8 case because G3 is not connected
    if np.all(np.isnan(Efd_g3)):
        continue

    plt.plot(
        results[n]["t"],
        Efd_g3,
        label=f"SCR={results[n]['SCR']:.1f}"
    )

plt.title("Field Voltage of G3")
plt.xlabel("Time [s]")
plt.ylabel("pu")
plt.legend()


# 4. VSC Reactive Power
plt.figure()
for n in results:

    Qv = np.array(results[n]["Q_vsc"]).squeeze()

    plt.plot(
        results[n]["t"],
        Qv,
        label=f"SCR={results[n]['SCR']:.1f}"
    )

plt.title("VSC Reactive Power")
plt.xlabel("Time [s]")
plt.ylabel("MVAr")
plt.legend()


plt.show() 