import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

# Create figure
fig, axes = plt.subplots(3, 1, figsize=(8, 6))

cases = [
    {
        "title": "Export Operation",
        "left": "Local System\nGeneration > Load",
        "right": "External Grid",
        "arrow": "right",
        "label": "Power Export",
        "description": "Surplus active power is transferred to the external grid"
    },
    {
        "title": "Balanced Operation",
        "left": "Local System\nGeneration ≈ Load",
        "right": "External Grid",
        "arrow": "both",
        "label": "Near-zero net exchange",
        "description": "Generation and demand are approximately balanced"
    },
    {
        "title": "Import Operation",
        "left": "Local System\nGeneration < Load",
        "right": "External Grid",
        "arrow": "left",
        "label": "Power Import",
        "description": "Additional active power is supplied from the external grid"
    }
]

for ax, case in zip(axes, cases):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    # Title
    ax.text(5, 2.75, case["title"], ha="center", va="center",
            fontsize=13, fontweight="bold")

    # Local system box
    local_box = Rectangle((0.7, 0.9), 2.4, 1.1, fill=False, linewidth=1.5)
    ax.add_patch(local_box)
    ax.text(1.9, 1.45, case["left"], ha="center", va="center", fontsize=10)

    # External grid box
    grid_box = Rectangle((6.9, 0.9), 2.4, 1.1, fill=False, linewidth=1.5)
    ax.add_patch(grid_box)
    ax.text(8.1, 1.45, case["right"], ha="center", va="center", fontsize=10)

    # Arrow
    if case["arrow"] == "right":
        arrow = FancyArrowPatch((3.3, 1.45), (6.7, 1.45),
                                arrowstyle="->", mutation_scale=18,
                                linewidth=1.8)
        ax.add_patch(arrow)

    elif case["arrow"] == "left":
        arrow = FancyArrowPatch((6.7, 1.45), (3.3, 1.45),
                                arrowstyle="->", mutation_scale=18,
                                linewidth=1.8)
        ax.add_patch(arrow)

    elif case["arrow"] == "both":
        arrow = FancyArrowPatch((3.3, 1.45), (6.7, 1.45),
                                arrowstyle="<->", mutation_scale=18,
                                linewidth=1.8)
        ax.add_patch(arrow)

    # Arrow label
    ax.text(5, 1.75, case["label"], ha="center", va="bottom",
            fontsize=10, fontweight="bold")

    # Short explanation
    ax.text(5, 0.45, case["description"], ha="center", va="center",
            fontsize=9)

plt.tight_layout()

# Save figure
plt.savefig("operating_conditions_combined.png", dpi=300, bbox_inches="tight")
plt.savefig("operating_conditions_combined.pdf", bbox_inches="tight")

plt.show()