# -*- coding: utf-8 -*-

import sys
import importlib
import numpy as np
import matplotlib.pyplot as plt

import tops.dynamic as dps
import tops.solvers as dps_sol

# --------------------------------------------------
# PATHS
# --------------------------------------------------
SRC_PATH = r"D:\Masters REM+\Master Thesis\paper\TOPS-main\TOPS-main\src"
EXAMPLE_PATH = r"D:\Masters REM+\Master Thesis\paper\TOPS-main\TOPS-main\examples\Base work"

sys.path.insert(0, SRC_PATH)
sys.path.insert(0, EXAMPLE_PATH)

# --------------------------------------------------
# IMPORT NETWORK
# --------------------------------------------------
import VSC_V_generator_network as model_data
importlib.reload(model_data)

model = model_data.load()

# --------------------------------------------------
# BUILD SYSTEM
# --------------------------------------------------
ps = dps.PowerSystemModel(model=model)

print("\nSystem created")

if hasattr(ps, "vsc"):
    print("✅ VSC detected:", list(ps.vsc.keys()))
else:
    print("❌ VSC NOT detected")

# --------------------------------------------------
# POWER FLOW + INIT
# --------------------------------------------------
ps.power_flow()
ps.init_dyn_sim()

# --------------------------------------------------
# SIMULATION SETUP
# --------------------------------------------------
SIM_TIME = 10.0

sol = dps_sol.ModifiedEulerDAE(
    ps.state_derivatives,
    ps.solve_algebraic,
    0.0,
    ps.x_0.copy(),
    SIM_TIME,
    max_step=2e-3
)

# --------------------------------------------------
# RESULTS STORAGE
# --------------------------------------------------
t_vals = []
V_B6_vals = []
Pvsc_vals = []

bus_idx = {'B1':0,'B2':1,'B5':2,'B6':3,'B7':4,'B8':5,'B9':6,'B10':7}
iB6 = bus_idx['B6']

t = 0.0
step_done = False

print("\nRunning simulation...")

while t < SIM_TIME:
    sol.step()

    t = sol.t
    x = sol.y
    v = sol.v

    # ----------------------------------------
    # STEP CHANGE (VSC control)
    # ----------------------------------------
    if (t > 1.0) and not step_done:
        ps.vsc['VSC_V'].set_input('P_setp', 600.0)   # export
        ps.vsc['VSC_V'].set_input('V_setp', 1.1)
        print(f"Step applied at t = {t:.2f}")
        step_done = True

    # ----------------------------------------
    # MEASUREMENTS
    # ----------------------------------------
    V_B6 = np.abs(v[iB6])
    Pvsc = ps.vsc['VSC_V'].P(x, v) * ps.sys_data['s_n']

    t_vals.append(t)
    V_B6_vals.append(V_B6)
    Pvsc_vals.append(Pvsc)

print("✅ Simulation finished")

# --------------------------------------------------
# PLOTS
# --------------------------------------------------
plt.figure()
plt.plot(t_vals, V_B6_vals)
plt.title("Voltage at B6")
plt.xlabel("Time [s]")
plt.ylabel("pu")
plt.grid()

plt.figure()
plt.plot(t_vals, Pvsc_vals)
plt.title("VSC Active Power")
plt.xlabel("Time [s]")
plt.ylabel("MW")
plt.grid()

plt.show()