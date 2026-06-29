import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Power-angle curves
# --------------------------------------------------

delta = np.linspace(0, np.pi, 800)

P_prefault  = 1.20 * np.sin(delta)
P_fault     = 0.35 * np.sin(delta)
P_postfault = 0.90 * np.sin(delta)

Pm = 0.65

# --------------------------------------------------
# Characteristic angles
# --------------------------------------------------

# Initial operating angle (Pm intersects pre-fault curve)
delta0 = np.arcsin(Pm / 1.20)

# Critical clearing angle
delta_c = 1.05

# Maximum rotor angle (Pm intersects post-fault curve)
delta_max = np.pi - np.arcsin(Pm / 0.90)

# --------------------------------------------------
# Plot
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(delta, P_prefault, lw=2.8, label='Pre-fault')
plt.plot(delta, P_fault, '--', lw=2.8, label='During fault')
plt.plot(delta, P_postfault, '-.', lw=2.8, label='Post-fault')

plt.axhline(Pm, color='C0', ls='--', lw=2, label=r'$P_m$')

# --------------------------------------------------
# Accelerating area A1
# --------------------------------------------------

dA1 = np.linspace(delta0, delta_c, 300)

plt.fill_between(
    dA1,
    Pm,
    0.35 * np.sin(dA1),
    color='steelblue',
    alpha=0.45
)

# --------------------------------------------------
# Decelerating area A2
# --------------------------------------------------

dA2 = np.linspace(delta_c, delta_max, 400)

plt.fill_between(
    dA2,
    0.90 * np.sin(dA2),
    Pm,
    color='#f6c667',
    alpha=0.75
)

# --------------------------------------------------
# Vertical markers
# --------------------------------------------------

for x in [delta0, delta_c, delta_max]:
    plt.axvline(x, color='k', ls=':', lw=2)

# --------------------------------------------------
# Labels
# --------------------------------------------------

plt.text(delta0, 0.03, r'$\delta_0$', ha='center', fontsize=14)
plt.text(delta_c, 0.03, r'$\delta_c$', ha='center', fontsize=14)
plt.text(delta_max, 0.03, r'$\delta_{max}$', ha='center', fontsize=14)

plt.text(0.80, 0.56, r'$A_1$', fontsize=20)
plt.text(1.60, 0.80, r'$A_2$', fontsize=20)

# --------------------------------------------------
# Axes
# --------------------------------------------------

plt.xlim(0, np.pi)
plt.ylim(0, 1.26)

plt.xlabel(r'Rotor angle, $\delta$ (rad)', fontsize=16)
plt.ylabel(r'Electrical power, $P$ (p.u.)', fontsize=16)

# Smaller title
plt.title(
    'Equal Area Criterion Illustrating Transient Stability and Critical Clearing Time',
    fontsize=16,
    pad=14
)

plt.legend(fontsize=11)

plt.tight_layout()

plt.savefig(
    'equal_area_criterion_cct.png',
    dpi=600,
    bbox_inches='tight'
)

plt.show()