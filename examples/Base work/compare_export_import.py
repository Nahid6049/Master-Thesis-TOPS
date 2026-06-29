# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import importlib

import tops.dynamic as dps
import tops.solvers as dps_sol

sys.path.append(r"D:\Masters REM+\Master Thesis\paper\TOPS-main\TOPS-main")
sys.path.append(r"D:\Masters REM+\Master Thesis\paper\TOPS-main\TOPS-main\examples\user_models")

import user_lib
import my_network as model_data


# ------------------------------------------------
# RUN ONE CASE
# ------------------------------------------------

def run_case(Pg3):

    importlib.reload(model_data)
    model = model_data.load()

    # modify G3 power
    for g in model['generators']['GEN'][1:]:
        if g[0] == 'G3':
            g[4] = Pg3

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

    t = 0

    while t < 10:

        x = sol.y
        v = sol.v

        # professor disturbance
        if t > 1:
            ps.gen['GEN'].v_setp(x, v)[1] = 1.1

        sol.step()

        t = sol.t
        v = sol.v

        res['t'].append(t)
        res['V_B6'].append(np.abs(v[iB6]))

    return res


# ------------------------------------------------
# RUN SCENARIOS
# ------------------------------------------------

export_case   = run_case(600)
balanced_case = run_case(300)
import_case   = run_case(100)


# ------------------------------------------------
# PLOT
# ------------------------------------------------

plt.figure()

plt.plot(export_case['t'], export_case['V_B6'], label="Export")
plt.plot(balanced_case['t'], balanced_case['V_B6'], label="Balanced")
plt.plot(import_case['t'], import_case['V_B6'], label="Import")

plt.title("Voltage Response at Bus B6")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (pu)")
plt.legend()
plt.grid()

plt.show()