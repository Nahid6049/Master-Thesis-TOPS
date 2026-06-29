import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Generate Q-V Curve
# ----------------------------

# Voltage range
V = np.linspace(0.2, 1.2, 600)

# Curve equation (smooth and realistic nose shape)
Q = 1.1 * V * (1 - V**1.4)

# Find nose point (maximum Q)
idx_nose = np.argmax(Q)
Q_nose = Q[idx_nose]
V_nose = V[idx_nose]

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(8,6))

# Main curve
plt.plot(Q, V, linewidth=2)

# Nose point
plt.scatter(Q_nose, V_nose)
plt.text(Q_nose - 0.2, V_nose,
         'Voltage Stability Limit',
         fontsize=10, va='center')

# Stable region (upper branch)
plt.text(0.05, 0.75, 'Stable Region', fontsize=10)

# Unstable region (lower branch)
plt.text(0.10, 0.25, 'Unstable Region', fontsize=10)

# Arrow (clean and short)
plt.annotate('Increasing Q demand',
             xy=(Q_nose - 0.01, V_nose + 0.02),
             xytext=(0.10, 1.00),   # moved slightly up
             arrowprops=dict(arrowstyle='->'),
             fontsize=10)

# Axis labels
plt.xlabel('Reactive Power Q (p.u.)')
plt.ylabel('Voltage V (p.u.)')

# Title
plt.title('Q–V Curve')



# Layout
plt.tight_layout()

# Show plot
plt.show()