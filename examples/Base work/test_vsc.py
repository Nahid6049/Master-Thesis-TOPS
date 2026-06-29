# -*- coding: utf-8 -*-

import sys
import importlib
import numpy as np
import matplotlib.pyplot as plt

import tops.dynamic as dps
import tops.solvers as dps_sol

BASE_PATH = r"D:\Masters REM+\Master Thesis\paper\TOPS-main\TOPS-main"
sys.path.insert(0, BASE_PATH + r"\examples\Base work")
sys.path.append(BASE_PATH + r"\examples\user_models")

import user_lib
import VSC_V_generator_network as model_data

importlib.reload(model_data)
model = model_data.load()

ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_lib)
ps.power_flow()
ps.init_dyn_sim()

vsc = ps.vsc['VSC_V']

sol = dps_sol.ModifiedEulerDAE(
    ps.state_derivatives,
    ps.solve_algebraic,
    0,
    ps.x_0.copy(),
    5.0,
    max_step=5e-3
)

t_hist = []
v_hist = []
q_hist = []

B8 = 5   # bus index of B8

while sol.t < 5.0:
    sol.step()

    if sol.t >= 1.0:
        vsc._input_values['V_setp'][:] = 1.02

    x = sol.y
    v = sol.v

    t_hist.append(sol.t)
    v_hist.append(abs(v[B8]))
    q_hist.append(float(np.asarray(vsc.Q(x, v)).reshape(-1)[0]))

print("Test ran successfully.")
print("Final B8 voltage =", v_hist[-1])
print("Final VSC Q =", q_hist[-1])

plt.figure()
plt.plot(t_hist, v_hist)
plt.xlabel("Time [s]")
plt.ylabel("B8 Voltage [pu]")
plt.title("VSC_V Test")

plt.figure()
plt.plot(t_hist, q_hist)
plt.xlabel("Time [s]")
plt.ylabel("VSC Reactive Power")
plt.title("VSC_V Reactive Power Response")

plt.show()