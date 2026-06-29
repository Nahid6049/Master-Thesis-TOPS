import sys
import importlib

# PATHS (same as yours)
sys.path.insert(0, r"D:\Masters REM+\Master Thesis\paper\TOPS-main\TOPS-main\examples\Base work")
sys.path.append(r"D:\Masters REM+\Master Thesis\paper\TOPS-main\TOPS-main\examples\user_models")

# 👇 IMPORTANT: use your file name
import update_gen_network as model_data


# ----------------------------------------
# SCR CALCULATION
# ----------------------------------------
def compute_scr(model, n):

    X_L56 = X_L25 = X_L69 = None
    X_T1 = X_T4 = None
    Xd_g1 = None
    Xd_local = None

    # Lines
    for line in model['lines'][1:]:
        if line[0] == 'L5-6':
            X_L56 = line[8]
        elif line[0] == 'L2-5':
            X_L25 = line[8]
        elif line[0] == 'L6-9':
            X_L69 = line[8]

    # Transformers
    for tr in model['transformers'][1:]:
        if tr[0] == 'T1':
            X_T1 = tr[7]
        elif tr[0] == 'T4':
            X_T4 = tr[7]

    # Generators
    for gen in model['generators']['GEN'][1:]:
        if gen[0] == 'G1':
            Xd_g1 = gen[12]
        elif gen[0] == 'G3':
            Xd_local = gen[12]

    # Upstream grid
    X_up = X_L56 + X_L25 + X_T1 + Xd_g1

    if n == 0:
        X_th = X_up
    else:
        X_loc = X_L69 + X_T4 + (Xd_local / n)
        X_th = (X_up * X_loc) / (X_up + X_loc)

    SCR = 1 / X_th

    return SCR


# ----------------------------------------
# MAIN
# ----------------------------------------
cases = [0, 1, 2, 4]

print("\n===== SCR RESULTS =====\n")

for n in cases:
    importlib.reload(model_data)
    model = model_data.load()

    scr = compute_scr(model, n)
    print(f"n = {n}  →  SCR = {scr:.3f}")

print("\nDone.\n")