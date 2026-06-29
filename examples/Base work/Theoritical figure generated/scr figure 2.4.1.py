import matplotlib.pyplot as plt

scr_values = [0, 2, 3, 6]
strength_levels = [1, 1, 1, 1]

plt.figure(figsize=(8, 2.8))

plt.axvspan(0, 2, alpha=0.25, label="Weak grid")
plt.axvspan(2, 3, alpha=0.25, label="Moderate grid")
plt.axvspan(3, 6, alpha=0.25, label="Strong grid")

plt.axvline(2, linestyle="--", linewidth=1.5)
plt.axvline(3, linestyle="--", linewidth=1.5)

plt.text(1, 0.55, "Weak grid\nSCR < 2", ha="center", fontsize=11)
plt.text(2.5, 0.55, "Moderate grid\n2 ≤ SCR ≤ 3", ha="center", fontsize=11)
plt.text(4.5, 0.55, "Strong grid\nSCR > 3", ha="center", fontsize=11)

plt.xlabel("Short Circuit Ratio (SCR)")
plt.yticks([])
plt.xlim(0, 6)
plt.ylim(0, 1)
plt.title("Grid Strength Classification Based on SCR")
plt.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("Figure_SCR_Classification.png", dpi=300, bbox_inches="tight")
plt.show()