import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# =========================================================
# Helper functions
# =========================================================
def draw_bus(ax, x, y, label, color="#9bd3e8"):
    ax.add_patch(Circle((x, y), 0.14, facecolor=color, edgecolor="black", lw=1.2))
    ax.plot([x, x], [y - 0.20, y + 0.20], color="black", lw=2)
    ax.text(x, y + 0.42, label, ha="center", va="bottom",
            fontsize=10, fontweight="bold")


def draw_generator_unit(ax, x, y, label, label_dx=0.62):
    ax.add_patch(Circle((x, y), 0.14, facecolor="#a6df71", edgecolor="black", lw=1.2))
    ax.text(x, y + 0.30, "G", ha="center", va="center",
            fontsize=11, fontweight="bold")
    ax.text(x + label_dx, y, label, ha="center", va="center", fontsize=9.2,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="gray", lw=1))


def draw_external_generator(ax, x, y, label_text):
    ax.add_patch(Circle((x, y), 0.14, facecolor="#a6df71", edgecolor="black", lw=1.2))
    ax.text(x, y + 0.30, "G", ha="center", va="center",
            fontsize=11, fontweight="bold")
    ax.text(x - 0.66, y, label_text, ha="center", va="center", fontsize=9.2,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="gray", lw=1))


def draw_vsc(ax, x, y):
    ax.add_patch(Circle((x, y), 0.16, facecolor="#b595e8", edgecolor="black", lw=1.2))
    ax.text(x, y - 0.32, "VSC", ha="center", va="center",
            fontsize=9, fontweight="bold")


def draw_label_box(ax, x, y, text):
    ax.text(x, y, text, ha="center", va="center",
            fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="gray", lw=1))


def draw_load(ax, x, y_bus, y_box, text):
    ax.annotate(
        "",
        xy=(x, y_box + 0.42),
        xytext=(x, y_bus - 0.20),
        arrowprops=dict(arrowstyle="-|>", lw=1.5, color="black")
    )
    ax.text(x, y_box, text, ha="center", va="center", fontsize=9.2,
            bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="gray", lw=1))


def draw_parallel_transformers(ax, B9, B10):
    left_bar_x = 8.25
    right_bar_x = 9.05

    ax.plot([left_bar_x, left_bar_x], [1.55, 3.05], color="black", lw=3)
    ax.plot([right_bar_x, right_bar_x], [1.55, 3.05], color="black", lw=3)

    ax.plot([B9[0], left_bar_x], [B9[1], B9[1]], color="black", lw=2)
    ax.plot([right_bar_x, B10[0]], [B10[1], B10[1]], color="black", lw=2)

    trafo_y = [2.85, 2.45, 2.05, 1.65]
    trafo_names = ["T4", "T5", "T6", "T7"]

    for y, name in zip(trafo_y, trafo_names):
        ax.plot([left_bar_x, 8.52], [y, y], color="black", lw=2)

        ax.text(8.65, y, name, ha="center", va="center",
                fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="gray", lw=1))

        ax.plot([8.78, right_bar_x], [y, y], color="black", lw=2)


# =========================================================
# Main drawing function
# =========================================================
def draw_system(ax, extended=False):
    ax.axis("off")
    ax.set_xlim(-1.1, 13.2)
    ax.set_ylim(-2.4, 4.7)

    title = (
        "Single-Line Diagram of the Studied System with Added Local Generators"
        if extended else
        "Single-Line Diagram of the Studied System"
    )

    ax.text(6.0, 4.25, title, ha="center", va="center",
            fontsize=16, fontweight="bold", family="serif")

    # Coordinates
    B1  = (0.8,  0.7)
    B2  = (2.65, 0.7)
    B5  = (4.25, 0.7)
    B6  = (5.85, 0.7)
    B9  = (7.65, 2.3)
    B10 = (9.35, 2.3)
    B7  = (7.65, -0.9)
    B8  = (9.35, -0.9)

    # Main network
    ax.plot([B1[0], B2[0]], [B1[1], B2[1]], color="black", lw=2)
    ax.plot([B2[0], B5[0]], [B2[1], B5[1]], color="black", lw=2)
    ax.plot([B5[0], B6[0]], [B5[1], B6[1]], color="black", lw=2)
    ax.plot([B6[0], B9[0]], [B6[1], B9[1]], color="black", lw=2)
    ax.plot([B6[0], B7[0]], [B6[1], B7[1]], color="black", lw=2)
    ax.plot([B7[0], B8[0]], [B7[1], B8[1]], color="black", lw=2)

    if not extended:
        ax.plot([B9[0], B10[0]], [B9[1], B10[1]], color="black", lw=2)

    ax.plot([0.2, B1[0]], [0.7, 0.7], color="black", lw=2)

    # Buses
    draw_bus(ax, *B1,  "B1\n20 kV")
    draw_bus(ax, *B2,  "B2\n420 kV")
    draw_bus(ax, *B5,  "B5\n420 kV")
    draw_bus(ax, *B6,  "B6\n420 kV")
    draw_bus(ax, *B9,  "B9\n420 kV")
    draw_bus(ax, *B10, "B10\n20 kV")
    draw_bus(ax, *B7,  "B7\n420 kV")
    draw_bus(ax, *B8,  "B8\n340 kV")

    # Generator and VSC
    draw_external_generator(ax, 0.2, 0.7, "GEN G1\nExternal Grid")
    draw_vsc(ax, B8[0], B8[1])

    # Local generation part
    if not extended:
        draw_label_box(ax, 8.50, 2.3, "T4")

        G3 = (10.6, 2.3)
        ax.plot([B10[0], G3[0] - 0.14], [B10[1], G3[1]], color="black", lw=2)
        draw_generator_unit(ax, *G3, "GEN G3")

    else:
        draw_parallel_transformers(ax, B9, B10)

        G3 = (11.0, 3.4)
        G4 = (11.0, 2.7)
        G5 = (11.0, 1.9)
        G6 = (11.0, 1.2)

        gen_bus_x = 10.2

        ax.plot([gen_bus_x, gen_bus_x], [1.0, 3.6], color="black", lw=2)
        ax.plot([B10[0], gen_bus_x], [B10[1], B10[1]], color="black", lw=2)

        for gy in [G3[1], G4[1], G5[1], G6[1]]:
            ax.plot([gen_bus_x, 10.85], [gy, gy], color="black", lw=2)

        draw_generator_unit(ax, *G3, "GEN G3")
        draw_generator_unit(ax, *G4, "GEN G4")
        draw_generator_unit(ax, *G5, "GEN G5")
        draw_generator_unit(ax, *G6, "GEN G6")

    # Labels
    draw_label_box(ax, 1.70, 0.7, "T1")
    draw_label_box(ax, 3.45, 0.7, "L2–5")
    draw_label_box(ax, 5.05, 0.7, "L5–6")
    draw_label_box(ax, 6.80, 1.60, "L6–9")
    draw_label_box(ax, 6.95, -0.10, "L6–7")
    draw_label_box(ax, 8.50, -0.9, "T3")

    # Loads
    draw_load(ax, B5[0], B5[1], -0.92, "L1\nP = 3000 MW\nQ = 100 MVAr")
    draw_load(ax, B6[0], B6[1], -0.92, "L2\nP = 800 MW\nQ = 50 MVAr")

    # System base note
    ax.text(9.05, -2.00, "System base: 1000 MVA, 50 Hz",
            ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="gray", lw=1))


# =========================================================
# Figure 1: Normal network
# =========================================================
fig1, ax1 = plt.subplots(figsize=(16, 8))
draw_system(ax1, extended=False)
plt.tight_layout()
fig1.savefig("single_line_normal_network.png", dpi=300, bbox_inches="tight")
fig1.savefig("single_line_normal_network.pdf", bbox_inches="tight")

# =========================================================
# Figure 2: Extended network
# =========================================================
fig2, ax2 = plt.subplots(figsize=(16, 8))
draw_system(ax2, extended=True)
plt.tight_layout()
fig2.savefig("single_line_extended_network.png", dpi=300, bbox_inches="tight")
fig2.savefig("single_line_extended_network.pdf", bbox_inches="tight")

plt.show()