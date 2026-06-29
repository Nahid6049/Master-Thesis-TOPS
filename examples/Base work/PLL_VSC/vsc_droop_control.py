import sys
import os

# Fix module path
current_dir = os.path.abspath(__file__)
tops_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
if tops_root not in sys.path:
    sys.path.insert(0, tops_root)

from collections import defaultdict
import matplotlib.pyplot as plt
import time
import numpy as np
import tops.dynamic as dps
import tops.solvers as dps_sol
import importlib
importlib.reload(dps)
import examples.user_models.user_lib as user_lib

if __name__ == '__main__':

    # Load model
    import tops.ps_models.k2a as model_data
    importlib.reload(model_data)
    model = model_data.load()

    model['pll'] = {'PLL1':[
        ['name', 'T_filter', 'bus'],
     #   *[[f'PLL{i}', 0.1, bus[0]] for i, bus in enumerate(model['buses'][1:])],
        ['PLL1', 0.1, 'B12'],
    ]}

    model['vsc'] = {'VSC': [
        ['name',    'T_pll',    'T_i',  'bus',  'P_K_p',    'P_K_i',    'Q_K_p',    'Q_K_i',    'P_setp',   'Q_setp'],
        # *[[f'VSC{i}', 0.1, 1, bus[0], 0.1, 0.1, 0.1, 0.1, 0.1, 0] for i, bus in enumerate(model['buses'][1:])],
        # "['VSC1',    0.1,        1,      'B12',   0.01,        1e-12,        0.1,        0.1,        0,          0],
        ['VSC1',      0.1,         1,   'B12',    0.1,      0.1,       0.1,        0.1,        100,          50],
    ]}

    # import dynpssimpy.user_models.user_lib as user_lib
    import examples.user_models.user_lib as user_lib

    # Power system model
    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_lib)
    ps.init_dyn_sim()
    print(max(abs(ps.ode_fun(0, ps.x_0))))

    x0 = ps.x_0
    v0 = ps.v_0

    t_end = 20
    x_0 = ps.x_0.copy()

    # Solver
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, t_end, max_step=5e-3)

    # Initialize simulation
    t = 0
    res = defaultdict(list)
    t_0 = time.time()
    Pcontrol = 100
    Qcontrol = 50
    ps.vsc['VSC'].set_input('P_setp', Pcontrol)
    ps.vsc['VSC'].set_input('Q_setp', Qcontrol)
    # Define variables to be stored
    Ivsc=0.0+0.0j
    Svsc=0.0+0.0j
    Svsc_stored=[]
    Igen=0.0+0.0j
    Strans=0.0+0.0j
    v7_stored = []
    P_e_stored = []
    Ptrans_stored = []
    frequency_stored = []
    tcount=0
    event_flag=True
    # Run simulation with droop control
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/(t_end)*100))
        # Load change on bus B9
        #if t > 10:
        # ps.y_bus_red_mod[8, 8] = 0.3

        # Line outage between bus B5 and B6 (line no. 1)
        # if t > 10 and event_flag:
        #    event_flag = False
        #    ps.lines['Line'].event(ps, ps.lines['Line'].par['name'][4], 'disconnect')

        # BESS control
        # if t > 2:
        #     Pcontrol = 50
        # if t > 5:
        #     Qcontrol = 50
        #
        ps.vsc['VSC'].set_input('P_setp', Pcontrol)
        ps.vsc['VSC'].set_input('Q_setp', Qcontrol)

        # Load change on bus B9
        if t > 10:
           ps.y_bus_red_mod[8, 8] = -0.1
        else:
           ps.y_bus_red_mod[8, 8] = 0

        # Short circuit on bus B7
        if 2 < t < 2.1:
            ps.y_bus_red_mod[6, 6] = 1000
        else:
            ps.y_bus_red_mod[6, 6] = 0

        # Simulate next step
        x = sol.y
        result = sol.step()
        t = sol.t
        v = sol.v

        dx = ps.ode_fun(0, ps.x_0)

        for mdl in ps.dyn_mdls:
            mdl.reset_outputs()

        # Compute power transfer B8-->B9
        Igen = ps.y_bus_red_full[6,7]*(v[7] -v[6])
        Strans=v[6]*np.conj(Igen)
        # Compute power transfer B12-->B8
        Ivsc = ps.y_bus_red_full[7, 11] * (v[7] - v[11])
        Svsc = v[7] * np.conj(Ivsc) * 900.0
        # Store result
        res['t'].append(sol.t)
        res['gen_speed'].append(ps.gen['GEN'].speed(x, v).copy())
        res['VSC_power'].append(ps.vsc['VSC'].P(x, v).copy())
        res['VSC_Q'].append(ps.vsc['VSC'].Q(x, v).copy())
        v7_stored.append(np.abs(v[7]))
        Ptrans_stored.append(np.real(Strans))
        Svsc_stored.append(Svsc)
        P_e_stored.append(ps.gen['GEN'].P_e(x, v).copy())
        SpeedVector= ps.gen['GEN'].speed(x, v)
        

        # frequency=50+50*0.25* (SpeedVector[0]+SpeedVector[1]+SpeedVector[2]+SpeedVector[3])
        #if t<5:
        #    frequency=50+50*0.25* (SpeedVector[0]+SpeedVector[1]+SpeedVector[2]+SpeedVector[3])
        #elif t>5:
        frequency=50+50*0.5* (SpeedVector[1]+SpeedVector[2])
        frequency_stored.append(frequency)

        # Proportional control of VSC P and Q
        # Pcontrol= 1000*(50-frequency)
        # Qcontrol= 500*(1 - abs(v[7]))
        # Qcontrol = 0
        #if t > 5:
        #    Pcontrol = 300
        tcount+=1
        
    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))
    print('Line outage', ps.lines['Line'].par['name'][4])
    # print('bus_ref_spec', ps.vsc['VSC'].bus_ref_spec)
    
    #Separating each generator speed
    genspeed1=[]
    genspeed2=[]
    genspeed3=[]
    genspeed4=[]
    p=0
    while p < tcount:
        genspeed1.append(res['gen_speed'][p][0])
        genspeed2.append(res['gen_speed'][p][1])
        genspeed3.append(res['gen_speed'][p][2])
        genspeed4.append(res['gen_speed'][p][3])
        p+=1
        
    # plot generator speeds
    plt.figure()
    plt.plot(res['t'], genspeed1, label='GEN1')
    plt.plot(res['t'], genspeed2, label='GEN2')
    plt.plot(res['t'], genspeed3, label='GEN3')
    plt.plot(res['t'], genspeed4, label='GEN4')
    plt.xlabel('Time [s]')
    plt.ylabel('Speed deviation [pu]')
    plt.legend()
    plt.ticklabel_format(useOffset=False, style='plain')
    # plt.show()

    # plot frequency as average speeds
    plt.figure()
    plt.plot(res['t'], np.array(frequency_stored))
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    fmin = np.argmax(frequency_stored)
    x_min = res['t'][fmin]
    y_min = frequency_stored[fmin]
    #plt.plot(x_min, y_min, marker='o')
    #plt.ticklabel_format(useOffset=False, style='plain')
    # plt.show()

    # Plot active power
    fig, ax = plt.subplots(2)
    fig.suptitle('Generator and VSC active power')
    ax[0].plot(res['t'], np.array(res['VSC_power']), res['t'], np.array(np.real(Svsc_stored)))
    ax[0].set_ylabel('VSC power (MW)')
    # ax[1].plot(res['t'], np.array(P_e_stored) / [900, 900, 900, 900])
    ax[1].plot(res['t'], np.array(P_e_stored))
    ax[1].set_ylabel('Gen. power (MW)')
    ax[1].set_xlabel('time (s)')

    # plot power transfer
    plt.figure()
    plt.plot(res['t'], np.array(Ptrans_stored))
    plt.xlabel('Time [s]')
    plt.ylabel('Power transfer [pu]')
    # plt.ticklabel_format(useOffset=False, style='plain')
    # plt.show()

    # plot voltage
    plt.figure()
    plt.plot(res['t'], np.array(v7_stored))
    plt.xlabel('Time [s]')
    plt.ylabel('Voltage @ B8 [pu]')
    #plt.ticklabel_format(useOffset=False, style='plain')
    #plt.show()

    plt.figure()
    plt.plot(res['t'], res['VSC_Q'], res['t'], np.array(np.imag(Svsc_stored)))
    plt.xlabel('Time [s]')
    plt.ylabel('Q_VSC [MVar]')
    #plt.ticklabel_format(useOffset=False, style='plain')
    #plt.legend()

    plt.show()
    