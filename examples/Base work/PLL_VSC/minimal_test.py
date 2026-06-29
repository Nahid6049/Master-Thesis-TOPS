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

tops_root = os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(current_dir)
                    )
                )
            )
if tops_root not in sys.path:
    sys.path.insert(0, tops_root)

import examples.user_models.user_lib as user_lib

base_work_dir = os.path.dirname(os.path.dirname(current_dir))
if base_work_dir not in sys.path:
    sys.path.insert(0, base_work_dir)

import my_network
model = my_network.load()

print("Available buses:")
print([b[0] for b in model['buses'][1:]])

# -------------------------------------------------
# ADD PLL
# -------------------------------------------------
model['pll'] = {
    'PLL1': [
        ['name', 'T_filter', 'bus'],
        ['PLL1', 0.1, 'B8'],
    ]
}

# -------------------------------------------------
# ADD VSC (correct professor header)
# -------------------------------------------------
model['vsc'] = {
    'VSC': [
        ['name','T_pll','T_i','bus',
         'P_K_p','P_K_i',
         'Q_K_p','Q_K_i',
         'P_setp','Q_setp'],
        ['VSC1',
         0.1,
         1.0,
         'B8',
         0.1, 0.1,
         0.1, 0.1,
         10, 5],   # P=10 MW, Q=5 MVar
    ]
}

# -------------------------------------------------
# BUILD SYSTEM
# -------------------------------------------------
ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_lib)

ps.power_flow()
ps.init_dyn_sim()

print("Initialization successful.")

# -------------------------------------------------
# SIMULATION
# -------------------------------------------------
t_end = 10.0
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
fpll_store = []

while sol.t < t_end:
    sol.step()

    x = sol.y
    v = sol.v

    t_store.append(sol.t)

    p_store.append(ps.vsc['VSC'].P(x, v)[0])
    q_store.append(ps.vsc['VSC'].Q(x, v)[0])

    f_est = ps.pll['PLL1'].freq_est(x, v)
    fpll_store.append(float(np.array(f_est).flatten()[0]))

print("Simulation completed.")

# -------------------------------------------------
# FINAL VALUES
# -------------------------------------------------
print("Final VSC P (MW):", p_store[-1])
print("Final VSC Q (MVar):", q_store[-1])
print("Final PLL freq (pu deviation):", fpll_store[-1])

# -------------------------------------------------
# PLOTS
# -------------------------------------------------
plt.figure()
plt.plot(t_store, p_store)
plt.title("VSC Active Power")
plt.xlabel("Time (s)")
plt.ylabel("P (MW)")
plt.grid(True)
plt.ylim(0, 12)

plt.figure()
plt.plot(t_store, q_store)
plt.title("VSC Reactive Power")
plt.xlabel("Time (s)")
plt.ylabel("Q (MVar)")
plt.grid(True)
plt.ylim(0, 12)

plt.figure()
plt.plot(t_store, fpll_store)
plt.title("PLL Frequency Estimate (pu deviation)")
plt.xlabel("Time (s)")
plt.ylabel("Δf (pu)")
plt.grid(True)
plt.ylim(0, 12)

plt.show()