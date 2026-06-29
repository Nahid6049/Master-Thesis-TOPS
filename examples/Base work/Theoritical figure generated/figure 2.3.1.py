import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

def draw_grid_figure(title, z_label, filename):
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(5, 5.3, title, ha="center", fontsize=14, fontweight="bold")

    source = Circle((1.5, 3), 0.6, fill=False, linewidth=2)
    ax.add_patch(source)
    ax.text(1.5, 3, r"$V_{th}$", ha="center", va="center", fontsize=12)

    ax.plot([2.1, 3.2], [3, 3], color="black", linewidth=2)

    z_block = Rectangle((3.2, 2.5), 1.6, 1.0, fill=False, linewidth=2)
    ax.add_patch(z_block)
    ax.text(4.0, 3, z_label, ha="center", va="center", fontsize=12)

    ax.plot([4.8, 6.5], [3, 3], color="black", linewidth=2)

    ax.plot(6.5, 3, "ko", markersize=5)
    ax.text(6.5, 3.35, "PCC", ha="center", fontsize=11)

    ax.plot([6.5, 7.2], [3, 3], color="black", linewidth=2)

    vsc = Rectangle((7.2, 2.3), 1.7, 1.4, fill=False, linewidth=2)
    ax.add_patch(vsc)
    ax.text(8.05, 3, "VSC", ha="center", va="center", fontsize=13)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


# Output 1: Strong grid
draw_grid_figure(
    title="Strong Grid",
    z_label=r"$Z_{th}$ small",
    filename="Figure_Strong_Grid.png"
)

# Output 2: Weak grid
draw_grid_figure(
    title="Weak Grid",
    z_label=r"$Z_{th}$ large",
    filename="Figure_Weak_Grid.png"
)