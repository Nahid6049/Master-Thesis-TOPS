import sys
import numpy as np
import matplotlib.pyplot as plt
import importlib
from copy import deepcopy

import tops.dynamic as dps
import tops.solvers as dps_sol

# PATHS
sys.path.append(r"D:\Masters REM+\Master Thesis\paper\TOPS-main\TOPS-main")
sys.path.append(r"D:\Masters REM+\Master Thesis\paper\TOPS-main\TOPS-main\examples\user_models")

import user_lib
import my_network as model_data


# -----------------------------------
# MODIFY GENERATORS (0,1,2,4)
# -----------------------------------
def modify_generators(model, n_local):

    model = deepcopy(model)

    gen_data = model['generators']['GEN']
    header = gen_data[0]
    gens = gen_data[1:]

    # get original G3
    original = None
    for g in gens:
        if g[0] == 'G3':
            original = g.copy()

    # remove G3
    gens = [g for g in gens if g[0] != 'G3']

    # clean controllers
    def clean(ctrl, key):
        header = ctrl[key][0]
        rows = ctrl[key][1:]
        rows = [r for r in rows if r[1] != 'G3']
        ctrl[key] = [header] + rows

    clean(model['gov'], 'HYGOV')
    clean(model['avr'], 'SEXS')
    clean(model['pss'], 'STAB1')

    # add generators
    new_gens, new_avr, new_gov, new_pss = [], [], [], []

    for i in range(n_local):
        name = f"G3_{i+1}" if i > 0 else "G3"

        g = original.copy()
        g[0] = name
        new_gens.append(g)

        new_avr.append(['AVR_'+name, name, 100, 0.5, 3.0, 0.1, -3, 3])
        new_gov.append(['HYGOV_'+name, name, 0.1, 1.5, 0.1, 2.0, 1, 1, 1, 0.01, 0.01, 0, 0.1, 1, 0])
        new_pss.append(['PSS_'+name, name, 50, 10.0, 0.5, 0.5, 0.05, 0.02, 0.03])

    model['generators']['GEN'] = [header] + gens + new_gens
    model['avr']['SEXS'] += new_avr
    model['gov']['HYGOV'] += new_gov
    model['pss']['STAB1'] += new_pss

    return model


# -----------------------------------
# RUN CASE (YOUR SOLVER STYLE)
# -----------------------------------
def run_case(n_local):

    importlib.reload(model_data)
    base_model = model_data.load()

    model = modify_generators(base_model, n_local)

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

    # bus index
    idx = {'B1':0,'B2':1,'B5':2,'B6':3,'B7':4,'B8':5,'B9':6,'B10':7}
    iB6 = idx['B6']

    t, v, q = [], [], []

    while sol.t < 10:
        sol.step()

        t.append(sol.t)

        # Voltage at Bus 6
        v.append(np.abs(ps.v[iB6]))

        # Reactive power of G3
        if n_local > 0:
            gen_names = ps.gen['GEN'].name
            idx_g = gen_names.index('G3')
            q.append(ps.gen['GEN'].Q[idx_g])
        else:
            q.append(0)

    return np.array(t), np.array(v), np.array(q)


# -----------------------------------
# RUN STUDY
# -----------------------------------
cases = {
    '0 gen': 0,
    '1 gen': 1,
    '2 gen': 2,
    '4 gen': 4,
}

results = {}

for name, n in cases.items():
    print(f"Running {name}...")
    results[name] = run_case(n)


# -----------------------------------
# PLOT
# -----------------------------------
plt.figure()
for name, (t, v, q) in results.items():
    plt.plot(t, v, label=name)
plt.title("Voltage at Bus 6")
plt.xlabel("Time [s]")
plt.ylabel("Voltage [p.u.]")
plt.legend()
plt.grid()

plt.figure()
for name, (t, v, q) in results.items():
    plt.plot(t, q, label=name)
plt.title("Reactive Power of G3")
plt.xlabel("Time [s]")
plt.ylabel("Q [p.u.]")
plt.legend()
plt.grid()

plt.show()