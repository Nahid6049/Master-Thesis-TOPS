# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import importlib
import tops.dynamic as dps
import tops.solvers as dps_sol

# paths
sys.path.append(r"D:\Masters REM+\Master Thesis\paper\TOPS-main\TOPS-main")
sys.path.append(r"D:\Masters REM+\Master Thesis\paper\TOPS-main\TOPS-main\examples\user_models")

import user_lib
import my_network as model_data


def run_case(Pg3):

    importlib.reload(model_data)
    model = model_data.load()

    # change generator power (export / import case)
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
    iB8 = idx['B8']

    res = defaultdict(list)

    t = 0

    while t < 10:

        x = sol.y
        v = sol.v

        # generator voltage step
        if t > 1:
            ps.gen['GEN'].v_setp(x, v)[1] = 1.1

        sol.step()

        t = sol.t
        x = sol.y
        v = sol.v

        V_B6 = np.abs(v[iB6])
        V_B8 = np.abs(v[iB8])

        Q_vsc = ps.vsc['VSC_SI'].q_e(x,v)*ps.sys_data['s_n']
        Q_gen = ps.gen['GEN'].q_e(x,v)*ps.sys_data['s_n']

        res['t'].append(t)
        res['V_B6'].append(V_B6)
        res['V_B8'].append(V_B8)
        res['Q_vsc'].append(Q_vsc)
        res['Q_gen'].append(Q_gen[1])

    return res


# scenarios
export_case   = run_case(600)
balanced_case = run_case(300)
import_case   = run_case(100)


# -------- PLOTS --------

# Voltage B6
plt.figure()
plt.plot(export_case['t'], export_case['V_B6'], label='Export')
plt.plot(balanced_case['t'], balanced_case['V_B6'], label='Balanced')
plt.plot(import_case['t'], import_case['V_B6'], label='Import')
plt.title("Voltage B6")
plt.legend()
plt.grid()

# Voltage B8
plt.figure()
plt.plot(export_case['t'], export_case['V_B8'], label='Export')
plt.plot(balanced_case['t'], balanced_case['V_B8'], label='Balanced')
plt.plot(import_case['t'], import_case['V_B8'], label='Import')
plt.title("Voltage B8")
plt.legend()
plt.grid()

# Generator Q
plt.figure()
plt.plot(export_case['t'], export_case['Q_gen'], label='Export')
plt.plot(balanced_case['t'], balanced_case['Q_gen'], label='Balanced')
plt.plot(import_case['t'], import_case['Q_gen'], label='Import')
plt.title("Generator Reactive Power")
plt.legend()
plt.grid()

# VSC Q
plt.figure()
plt.plot(export_case['t'], export_case['Q_vsc'], label='Export')
plt.plot(balanced_case['t'], balanced_case['Q_vsc'], label='Balanced')
plt.plot(import_case['t'], import_case['Q_vsc'], label='Import')
plt.title("VSC Reactive Power")
plt.legend()
plt.grid()

plt.show() 
