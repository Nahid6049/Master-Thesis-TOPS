import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(18, 9))

# -------------------------------------------------
# BUS POSITIONS
# -------------------------------------------------
B1  = (0, 0)
B2  = (2, 0)
B5  = (4, 0)
B6  = (6, 0)

B7  = (8, -2)
B8  = (10, -2)

B9  = (8, 2)
B10 = (12, 2)

# -------------------------------------------------
# DRAW BUS
# -------------------------------------------------
def draw_bus(x, y, name, kv, color="skyblue"):
    ax.scatter(
        x, y,
        s=320,
        color=color,
        edgecolor="black",
        zorder=5
    )

    ax.plot([x - 0.12, x + 0.12], [y, y], "k", lw=1.2)
    ax.plot([x, x], [y - 0.12, y + 0.12], "k", lw=1.2)

    ax.text(
        x,
        y + 0.55,
        f"{name}\n{kv} kV",
        ha="center",
        fontsize=10,
        fontweight="bold"
    )

# -------------------------------------------------
# MAIN NETWORK
# -------------------------------------------------
ax.plot([0, 2], [0, 0], "k", lw=2)
ax.plot([2, 4], [0, 0], "k", lw=2)
ax.plot([4, 6], [0, 0], "k", lw=2)

ax.plot([6, 8], [0, 2], "k", lw=2)
ax.plot([6, 8], [0, -2], "k", lw=2)

ax.plot([8, 10], [-2, -2], "k", lw=2)

# -------------------------------------------------
# PARALLEL TRANSFORMER BANK T4-T7
# -------------------------------------------------
# B9 to left busbar
ax.plot([8.0, 8.8], [2.0, 2.0], "k", lw=2)

# left and right transformer busbars
ax.plot([8.8, 8.8], [1.1, 2.9], "k", lw=3)
ax.plot([10.8, 10.8], [1.1, 2.9], "k", lw=3)

yvals = [2.6, 2.2, 1.8, 1.4]
tnames = ["T4", "T5", "T6", "T7"]

for y, tname in zip(yvals, tnames):
    # left conductor
    ax.plot([8.8, 9.35], [y, y], "k", lw=2)

    # transformer rectangle
    rect = plt.Rectangle(
        (9.35, y - 0.12),
        0.65,
        0.24,
        fill=False,
        edgecolor="black",
        linewidth=1.5
    )
    ax.add_patch(rect)

    # transformer label inside rectangle
    ax.text(
        9.675,
        y,
        tname,
        fontsize=8,
        ha="center",
        va="center"
    )

    # right conductor
    ax.plot([10.0, 10.8], [y, y], "k", lw=2)

# transformer bank to B10
ax.plot([10.8, 12.0], [2.0, 2.0], "k", lw=2)

# -------------------------------------------------
# T1 / T3 LABELS
# -------------------------------------------------
ax.text(
    1, 0,
    "T1",
    fontsize=8,
    ha="center",
    va="center",
    bbox=dict(fc="white", ec="gray")
)

ax.text(
    9, -2,
    "T3",
    fontsize=8,
    ha="center",
    va="center",
    bbox=dict(fc="white", ec="gray")
)

# -------------------------------------------------
# BUSES
# -------------------------------------------------
draw_bus(*B1, "B1", 20)
draw_bus(*B2, "B2", 420)
draw_bus(*B5, "B5", 420)
draw_bus(*B6, "B6", 420)

draw_bus(*B7, "B7", 420)
draw_bus(*B9, "B9", 420)
draw_bus(*B10, "B10", 20)

draw_bus(*B8, "B8", 340, color="mediumpurple")

# -------------------------------------------------
# LINE LABELS
# -------------------------------------------------
ax.text(
    3, 0,
    "L2-5",
    fontsize=8,
    ha="center",
    va="center",
    bbox=dict(fc="white", ec="gray")
)

ax.text(
    5, 0,
    "L5-6",
    fontsize=8,
    ha="center",
    va="center",
    bbox=dict(fc="white", ec="gray")
)

ax.text(
    7, 1.0,
    "L6-9",
    fontsize=8,
    ha="center",
    va="center",
    bbox=dict(fc="white", ec="gray")
)

ax.text(
    7, -1.0,
    "L6-7",
    fontsize=8,
    ha="center",
    va="center",
    bbox=dict(fc="white", ec="gray")
)

# -------------------------------------------------
# LOADS
# -------------------------------------------------
ax.arrow(
    4, -0.05,
    0, -1.2,
    head_width=0.08,
    length_includes_head=True
)

ax.text(
    4, -2.0,
    "L1\nP = 3000 MW\nQ = 100 MVAr",
    fontsize=9,
    ha="center",
    bbox=dict(fc="white", ec="gray")
)

ax.arrow(
    6, -0.05,
    0, -1.2,
    head_width=0.08,
    length_includes_head=True
)

ax.text(
    6, -2.0,
    "L2\nP = 800 MW\nQ = 50 MVAr",
    fontsize=9,
    ha="center",
    bbox=dict(fc="white", ec="gray")
)

# -------------------------------------------------
# EXTERNAL GRID
# -------------------------------------------------
ax.plot([-0.6, 0], [0, 0], "k", lw=2)

ax.scatter(
    -0.6,
    0,
    s=320,
    color="yellowgreen",
    edgecolor="black",
    zorder=5
)

ax.text(
    -0.95,
    0.45,
    "G",
    fontsize=11,
    fontweight="bold"
)

ax.text(
    -1.45,
    -0.05,
    "GEN G1\nExternal Grid",
    fontsize=9,
    ha="center",
    va="center",
    bbox=dict(fc="white", ec="gray")
)

# -------------------------------------------------
# LOCAL GENERATOR COLLECTOR
# -------------------------------------------------
# connection from B10 to collector bus
ax.plot([12, 13], [2, 2], "k", lw=2)

# collector bus
ax.plot(
    [13, 13],
    [0.6, 3],
    "k",
    lw=2
)

gen_y = [3.0, 2.2, 1.4, 0.6]
gen_n = ["G3", "G4", "G5", "G6"]

for y, gname in zip(gen_y, gen_n):
    ax.plot(
        [13, 13.8],
        [y, y],
        "k",
        lw=2
    )

    ax.scatter(
        14.0,
        y,
        s=320,
        color="yellowgreen",
        edgecolor="black",
        zorder=5
    )

    ax.text(
        13.95,
        y + 0.45,
        "G",
        fontsize=11,
        fontweight="bold"
    )

    ax.text(
        14.45,
        y,
        f"GEN {gname}",
        fontsize=8,
        va="center",
        bbox=dict(fc="white", ec="gray")
    )

# -------------------------------------------------
# VSC
# -------------------------------------------------
ax.text(
    10,
    -2.45,
    "VSC",
    ha="center",
    fontsize=10,
    fontweight="bold"
)



# -------------------------------------------------
# FOOTNOTE
# -------------------------------------------------
ax.text(
    8.6,
    -3.5,
    "System base: 1000 MVA, 50 Hz",
    fontsize=9,
    bbox=dict(fc="white", ec="gray")
)

# -------------------------------------------------
# TITLE
# -------------------------------------------------
ax.set_title(
    "Single-Line Diagram of the Studied System with Added Local Generators",
    fontsize=18,
    fontweight="bold"
)

ax.axis("off")
ax.set_xlim(-2, 17)
ax.set_ylim(-4, 5)

plt.tight_layout()
plt.show()