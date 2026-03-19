import numpy as np
import matplotlib.pyplot as plt

def fermi_filter(r, r_zero, width):
    """Fungsi Filter Fermi"""
    return 1 / (1 + np.exp((r - r_zero) / width))

# Setup Parameter Matriks 320 (setengah lebar = 160)
N_half = 160
r0 = 140
w = 12

# Membuat sumbu k-space (jarak dari pusat)
r = np.linspace(0, N_half, 500)
f_weight = fermi_filter(r, r0, w)

# Plotting
plt.plot(r, f_weight, label=f'r0={r0}, w={w}')
plt.title("Profil Filter Fermi k-space (320x320)")
plt.xlabel("Radius k-space (pixel)")
plt.ylabel("Bobot")
plt.axvline(x=r0, color='red', linestyle='--', label='Cutoff Radius')
plt.legend()
plt.grid(True)
plt.show()
