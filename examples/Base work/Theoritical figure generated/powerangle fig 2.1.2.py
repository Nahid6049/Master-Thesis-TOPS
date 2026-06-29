import numpy as np
import matplotlib.pyplot as plt

delta = np.linspace(0, np.pi, 100)
Pmax = 1
P = Pmax * np.sin(delta)

plt.figure()
plt.plot(delta * 180/np.pi, P)
plt.axvline(x=90, linestyle='--')
plt.xlabel("Rotor Angle δ (degrees)")
plt.ylabel("Power P (p.u.)")
plt.title("Power-Angle Curve")

plt.show()