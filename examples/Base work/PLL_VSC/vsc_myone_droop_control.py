# -*- coding: utf-8 -*-
"""
RjKvill Network
VSC_SI + PLL Interaction
Generator Electrical Power Added
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


if __name__ == '__main__':

    # ==========================================================
    # LOAD NETWORK
    # ==========================================================
    import my_network as model_data
    importlib.reload(model_data)
    model = model_data.load()

    import examples.user_models.user_lib as user_lib

    # ==========================================================
    # ADD PLL AT BUS B8
    # ==========================================================
    model['pll'] = {'PLL1': [
        ['name', 'T_filter', 'bus'],
        ['PLL1', 0.1, 'B8'],
    ]}

    # ==========================================================
    # INITIALIZE SYSTEM
    # ==========================================================
    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_lib)
    ps.init_dyn_sim()

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

        # Load change at B6
        if t > 10:
            ps.y_bus_red_mod[iB6, iB6] = -0.1
        else:
            ps.y_bus_red_mod[iB6, iB6] = 0

        # Fault at B7 (2–2.1 s)
        if 2 < t < 2.1:
            ps.y_bus_red_mod[iB7, iB7] = 1000
        else:
            ps.y_bus_red_mod[iB7, iB7] = 0

        # Solver step
        x = sol.y
        sol.step()
        t = sol.t
        v = sol.v

        for mdl in ps.dyn_mdls:
            mdl.reset_outputs()

        # ======================================================
        # 1️⃣ Generator speed
        # ======================================================
        SpeedVector = ps.gen['GEN'].speed(x, v)
        frequency = 50 + 50*np.mean(SpeedVector)

        # ======================================================
        # 2️⃣ Generator electrical power (ADDED)
        # ======================================================
        P_gen = ps.gen['GEN'].P_e(x, v)   # MW already in TOPS

        # ======================================================
        # 3️⃣ PLL frequency
        # ======================================================
        pll_freq = ps.pll['PLL1'].freq_est(x, v)

        # ======================================================
        # 4️⃣ VSC active power
        # ======================================================
        P_vsc = ps.vsc['VSC_SI'].p_e(x, v) * ps.sys_data['s_n']

        # ======================================================
        # 5️⃣ VSC reactive power
        # ======================================================
        Q_vsc = ps.vsc['VSC_SI'].q_e(x, v) * ps.sys_data['s_n']

        # ======================================================
        # 6️⃣ Line power B6 → B7
        # ======================================================
        I_line = ps.y_bus_red_full[iB6, iB7] * (v[iB6] - v[iB7])
        S_line = v[iB6] * np.conj(I_line) * ps.sys_data['s_n']

        # ======================================================
        # 7️⃣ Bus voltage B8
        # ======================================================
        V_B8 = np.abs(v[iB8])

        # ======================================================
        # STORE
        # ======================================================
        res['t'].append(t)
        res['gen_speed'].append(SpeedVector.copy())
        res['P_gen'].append(P_gen.copy())   # ADDED
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

   # ==========================================================
# PLOTS (THESIS READY FORMAT)
# ==========================================================

# 1️⃣ Generator Speeds
plt.figure()
plt.plot(res['t'], genspeed1, label='GEN1')
plt.plot(res['t'], genspeed2, label='GEN2')
plt.xlabel('Time [s]')
plt.ylabel('Speed deviation [pu]')
plt.legend()
plt.title("Generator Speed Response",fontsize=10)

# 2️⃣ Generator Electrical Power
plt.figure()
plt.plot(res['t'], P_g1, label='GEN1')
plt.plot(res['t'], P_g2, label='GEN2')
plt.xlabel('Time [s]')
plt.ylabel('Generator Electrical Power [MW]')
plt.legend()
plt.title("Generator Electrical Power Response",fontsize=10)

# 3️⃣ System Frequency
plt.figure()
plt.plot(res['t'], res['frequency'])
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.title("System Frequency Response",fontsize=10)

# 4️⃣ PLL Frequency
plt.figure()
plt.plot(res['t'], 50+50*np.array(res['pll_freq']))
plt.xlabel('Time [s]')
plt.ylabel('PLL Estimated Frequency [Hz]')
plt.title("PLL Frequency Tracking",fontsize=10)

# 5️⃣ VSC Active Power
plt.figure()
plt.plot(res['t'], res['P_vsc'])
plt.xlabel('Time [s]')
plt.ylabel('VSC Active Power [MW]')
plt.title("VSC Active Power Response",fontsize=10)

# 6️⃣ VSC Reactive Power
plt.figure()
plt.plot(res['t'], res['Q_vsc'])
plt.xlabel('Time [s]')
plt.ylabel('VSC Reactive Power [MVar]')
plt.title("VSC Reactive Power Response",fontsize=10)

# 7️⃣ Line Power
plt.figure()
plt.plot(res['t'], res['P_line'])
plt.xlabel('Time [s]')
plt.ylabel('Line Power B6 → B7 [MW]')
plt.title("Transmission Line Power Transfer",fontsize=10)

# 8️⃣ Bus Voltage
plt.figure()
plt.plot(res['t'], res['V_B8'])
plt.xlabel('Time [s]')
plt.ylabel('Voltage @ B8 [pu]')
plt.title("Bus Voltage Response at B8",fontsize=10)

plt.show()