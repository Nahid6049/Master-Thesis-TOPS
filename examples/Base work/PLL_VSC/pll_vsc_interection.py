
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import tops.dynamic as dps
import tops.solvers as dps_sol

# -------------------------------------------------
# PATH FIX
# -------------------------------------------------
current_dir = os.path.abspath(__file__)
tops_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
if tops_root not in sys.path:
    sys.path.insert(0, tops_root)

import examples.user_models.user_lib as user_lib

base_work_dir = os.path.dirname(os.path.dirname(current_dir))
if base_work_dir not in sys.path:
    sys.path.insert(0, base_work_dir)

import my_network
model = my_network.load()

# -------------------------------------------------
# SIMPLE VSC (NO VSC_SI)
# -------------------------------------------------
model['vsc'] = {
    'VSC': [
        ['name','T_pll','T_i','bus',
         'P_K_p','P_K_i',
         'Q_K_p','Q_K_i',
         'P_setp','Q_setp'],

        ['VSC1',
         0.5,      # slower PLL
         0.2,      # slower current dynamics
         'B8',

         0.2, 1.0,   # much smaller P gains
         0.2, 1.0,   # much smaller Q gains

         50.0,       # smaller power (50 MW instead of 200)
         0.0]
    ]
}

# -------------------------------------------------
# SIMULATION SETTINGS
# -------------------------------------------------
t_end = 20.0
sol = dps_sol.ModifiedEulerDAE(
    ps.state_derivatives,
    ps.solve_algebraic,
    0,
    ps.x_0.copy(),
    t_end,
    max_step=5e-3
)

t_store = []
p_store = []
q_store = []
pll_angle_store = []

event_done = False
 
# -------------------------------------------------
# SIMULATION LOOP
# -------------------------------------------------
while sol.t < t_end:

    sol.step()
    x, v, t = sol.y, sol.v, sol.t
# simulate voltage setpoint change
if t > 1:
    ps.gen['GEN'].v_setp(x,v)[1] = 1.1
    # Load increase at 10 s
    if (t > 10.0) and (not event_done):
        ps.loads['Load'].par['P'][0] *= 1.1
        ps.loads['Load'].par['Q'][0] *= 1.1
        event_done = True
        print("Applied +10% load step at B5 at t≈", t)

    t_store.append(t)

    # VSC power
    p_store.append(vsc.P(x, v)[0])
    q_store.append(vsc.Q(x, v)[0])

    # PLL angle directly (no freq_est)
    pll_angle = vsc.pll.output(x, v)[0]
    pll_angle_store.append(pll_angle)

print("Simulation completed.")
print("Final P (MW):", p_store[-1])
print("Final Q (MVAr):", q_store[-1])

# -------------------------------------------------
# PLOTS
# -------------------------------------------------
plt.figure()
plt.plot(t_store, p_store)
plt.title("VSC Active Power (MW)")
plt.grid(True)

plt.figure()
plt.plot(t_store, q_store)
plt.title("VSC Reactive Power (MVAr)")
plt.grid(True)

plt.figure()
plt.plot(t_store, pll_angle_store)
plt.title("PLL Angle (rad)")
plt.grid(True)

plt.show()