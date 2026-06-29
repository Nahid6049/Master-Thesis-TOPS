import sys
import os
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
tops_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.insert(0, tops_root)

import my_network
import user_lib
from tops.dynamic import PowerSystemModel
import tops.solvers as dps_sol

model = my_network.load()
ps = PowerSystemModel(model=model, user_mdl_lib=user_lib)
ps.init_dyn_sim()

vsc = ps.vsc["VSC"]
b8_idx = vsc.bus_idx_red["terminal"][0]

print("Initial P:", vsc.P(ps.x_0, ps.v_0)[0])

t_end = 5
solver = dps_sol.ModifiedEulerDAE(
    ps.state_derivatives,
    ps.solve_algebraic,
    0,
    ps.x_0.copy(),
    t_end,
    max_step=5e-3
)

t_list = []
P_list = []
Q_list = []
angle_list = []

t = 0
while t < t_end:
    solver.step()
    t = solver.t
    x = solver.y.copy()
    v = ps.solve_algebraic(t, x)

    t_list.append(t)
    P_list.append(vsc.P(x, v)[0])
    Q_list.append(vsc.Q(x, v)[0])
    angle_list.append(vsc.pll.output(x, v)[0])

plt.figure()
plt.plot(t_list, P_list)
plt.title("Active Power")

plt.figure()
plt.plot(t_list, Q_list)
plt.title("Reactive Power")

plt.figure()
plt.plot(t_list, angle_list)
plt.title("PLL Angle")

plt.show()