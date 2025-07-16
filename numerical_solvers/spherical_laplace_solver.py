import numpy as np
from scipy.special import sph_harm_y
import plotly.graph_objects as go
from scipy.interpolate import griddata

# Promień wewnętrzny i zewnętrzny sfery
a = 0.2      # np. 1 metr
R = 1.0      # np. 2 metry

# Liczba punktów siatki w każdym wymiarze
Nr = 50     # liczba punktów w kierunku radialnym
Ntheta = 50 # liczba punktów w kierunku θ
Nphi = 50   # liczba punktów w kierunku φ

r = np.linspace(a, R, Nr)
theta = np.linspace(0, np.pi, Ntheta)
phi = np.linspace(0, 2 * np.pi, Nphi, endpoint=False)

# Siatka 3D: r, theta, phi – użyjemy 'ij' żeby zachować porządek (r, θ, φ)
R_grid, Theta_grid, Phi_grid = np.meshgrid(r, theta, phi, indexing='ij')
V = np.full((Nr, Ntheta, Nphi), np.nan)  




# Zastosowanie warunku brzegowego na zewnętrznej sferze r = R
V[-1, :, :] = np.sin(5 * Phi_grid[-1, :, :])
C = 1.0  # potencjał na wewnętrznej kuli
V[0, :, :] = C



L_max = 10  # maksymalny rząd rozwinięcia
c_lm = {}   # współczynniki c_{lm}

# Wartość funkcji brzegowej na sferze r = R
f_theta_phi = np.sin(5 * Phi_grid[-1, :, :])  # shape: (Ntheta, Nphi)

# Siatka do całkowania (theta, phi)
dtheta = theta[1] - theta[0]
dphi = phi[1] - phi[0]
theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')  # shape: (Ntheta, Nphi)

# Oblicz współczynniki c_{lm}
for l in range(L_max + 1):
    for m in range(-l, l + 1):
        Y_lm = sph_harm_y(m, l, phi_grid, theta_grid)  # shape: (Ntheta, Nphi)
        integrand = f_theta_phi * np.conj(Y_lm) * np.sin(theta_grid)
        c = np.sum(integrand) * dtheta * dphi
        c_lm[(l, m)] = c





A_lm = {}
B_lm = {}

for (l, m), c in c_lm.items():
    if l == 0 and m == 0:
        # Układ równań dla l=0, m=0
        A = np.array([[1, a**(-1)],
                      [1, R**(-1)]])
        b = np.array([C, c])
        A00, B00 = np.linalg.solve(A, b)
        A_lm[(0, 0)] = A00
        B_lm[(0, 0)] = B00
    else:
        denom = R**l - a**(2*l + 1) * R**(-l - 1)
        if denom == 0:
            A_val = 0
        else:
            A_val = c / denom
        B_val = -A_val * a**(2*l + 1)
        A_lm[(l, m)] = A_val
        B_lm[(l, m)] = B_val


for l in range(L_max + 1):
    for m in range(-l, l + 1):
        A = A_lm[(l, m)]
        B = B_lm[(l, m)]

        # Część radialna
        radial_part = A * R_grid**l + B * R_grid**(-(l + 1))

        # Harmonica sferyczna
        Y_lm = sph_harm_y(m, l, Phi_grid, Theta_grid)

        # Dodaj wkład do potencjału – tylko rzeczywista część
        V += np.real(radial_part * Y_lm)





V_boundary_numeric = V[-1, :, :]  # r = R
V_boundary_expected = np.sin(5 * Phi_grid[-1, :, :])

max_diff = np.max(np.abs(V_boundary_numeric - V_boundary_expected))
print("Maksymalny błąd na zewnętrznej sferze:", max_diff)


# Środek siatki
i = Nr // 2
j = Ntheta // 2
k = Nphi // 2

laplace_approx = (
    V[i+1, j, k] + V[i-1, j, k] +
    V[i, j+1, k] + V[i, j-1, k] +
    V[i, j, k+1] + V[i, j, k-1] -
    6 * V[i, j, k]
)

print("Przybliżona Laplasjana w środku sfery:", laplace_approx)


import matplotlib.pyplot as plt

# Przekrój r–phi w płaszczyźnie θ = π/2 (środkowy indeks)
theta_idx = Ntheta // 2

plt.figure(figsize=(8, 4))
plt.imshow(V[:, theta_idx, :], extent=(0, 2*np.pi, a, R), aspect='auto', cmap='viridis')
plt.colorbar(label='V(r, θ=π/2, φ)')
plt.xlabel('φ')
plt.ylabel('r')
plt.title('Przekrój potencjału w płaszczyźnie θ=π/2')
plt.show()
