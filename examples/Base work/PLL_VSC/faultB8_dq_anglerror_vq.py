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
    

        # Fault at B8
        if 1 < t < 1.1:
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
    # ==========================================================
# PLL–VSC INTERACTION PLOT
# ==========================================================
fig, ax = plt.subplots(3,1,sharex=True)

# PLL phase error
ax[0].plot(t_vec, res['Vq'], color='green')
ax[0].axvline(1, color='red', linestyle='--')
ax[0].axvline(1.1, color='red', linestyle='--')
ax[0].set_ylabel("Vq - PLL Phase error")
ax[0].set_title("PLL–VSC Interaction")


# dq current response
ax[1].plot(t_vec, res['i_d'], color='green', label='i_d')
ax[1].plot(t_vec, res['i_q'], color='orange', label='i_q')
ax[1].axvline(1, color='red', linestyle='--')
ax[1].axvline(1.1, color='red', linestyle='--')
ax[1].set_ylabel("Current [pu]")
ax[1].legend()


# VSC power response
ax[2].plot(t_vec, res['P_vsc'], color='green', label='P_vsc')
ax[2].plot(t_vec, res['Q_vsc'], color='orange', label='Q_vsc')
ax[2].axvline(1, color='red', linestyle='--')
ax[2].axvline(1.1, color='red', linestyle='--')
ax[2].set_ylabel("Power [MW]")
ax[2].set_xlabel("Time [s]")
ax[2].legend()

plt.tight_layout()
plt.show()