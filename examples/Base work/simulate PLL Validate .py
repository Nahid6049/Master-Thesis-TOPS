# -*- coding: utf-8 -*-
"""
PLL Validation Study
PLL Frequency Estimate vs System Frequency
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


# ---------------------------------------------------------
# USER LIBRARY
# ---------------------------------------------------------
import user_models.user_lib as user_lib


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------
T_PLL = 0.05

T_END = 20.0
MAX_STEP = 5e-3

FAULT_START = 1.0
FAULT_END = 1.05

LOAD_STEP_TIME = 10.0

FAULT_BUS = "B8"

F_NOM = 50.0


# ---------------------------------------------------------
# MAIN SIMULATION
# ---------------------------------------------------------
def run_simulation():

    import my_network as model_data
    importlib.reload(model_data)

    model = model_data.load()

    # -----------------------------------------------------
    # Set PLL time constant
    # -----------------------------------------------------
    model["vsc"]["VSC_SI"][1][10] = T_PLL

    # -----------------------------------------------------
    # Create power system model
    # -----------------------------------------------------
    ps = dps.PowerSystemModel(
        model=model,
        user_mdl_lib=user_lib
    )

    ps.init_dyn_sim()

    # -----------------------------------------------------
    # Solver
    # -----------------------------------------------------
    sol = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0,
        ps.x_0.copy(),
        T_END,
        max_step=MAX_STEP
    )

    # -----------------------------------------------------
    # Bus indices
    # -----------------------------------------------------
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
    iFault = idx[FAULT_BUS]

    # -----------------------------------------------------
    # Storage
    # -----------------------------------------------------
    res = defaultdict(list)

    # -----------------------------------------------------
    # Time simulation loop
    # -----------------------------------------------------
    while sol.t < T_END:

        t = sol.t

     

        # -------------------------------------------------
        # Three-phase fault at B8
        # -------------------------------------------------
        if FAULT_START < t < FAULT_END:
            ps.y_bus_red_mod[iFault, iFault] = 1000
        else:
            ps.y_bus_red_mod[iFault, iFault] = 0.0

        # -------------------------------------------------
        # Solve step
        # -------------------------------------------------
        x = sol.y

        sol.step()

        t = sol.t
        v = sol.v

        # -------------------------------------------------
        # Reset outputs
        # -------------------------------------------------
        for mdl in ps.dyn_mdls:
            mdl.reset_outputs()

        # -------------------------------------------------
        # SYSTEM FREQUENCY
        # -------------------------------------------------
        speed_vector = ps.gen["GEN"].speed(x, v)

        f_sys = F_NOM + F_NOM * np.mean(speed_vector)

        # -------------------------------------------------
        # PLL FREQUENCY ESTIMATE
        # -------------------------------------------------
        f_pll = ps.vsc["VSC_SI"].freq_est(x, v)

        f_pll = float(np.asarray(f_pll).flatten()[0])

        # -------------------------------------------------
        # Store results
        # -------------------------------------------------
        res["t"].append(t)
        res["f_sys"].append(float(f_sys))
        res["f_pll"].append(float(f_pll))

    return res


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":

    res = run_simulation()

    t = np.array(res["t"])
    f_sys = np.array(res["f_sys"])
    f_pll = np.array(res["f_pll"])

    # -----------------------------------------------------
    # Plot
    # -----------------------------------------------------
    plt.style.use("default")

    fig = plt.figure(figsize=(8, 4), facecolor="white")

    plt.plot(
        t,
        f_sys,
        linewidth=2,
        label="System Frequency"
    )

    plt.plot(
        t,
        f_pll,
        "--",
        linewidth=2,
        label="PLL Frequency Estimate"
    )

    plt.title("PLL Frequency Estimate vs System Frequency")

    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")

    plt.legend()

    ax = plt.gca()
    ax.set_facecolor("white")

    plt.grid(False)

    plt.tight_layout()

    plt.show()