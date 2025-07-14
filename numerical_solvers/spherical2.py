import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.special import legendre

# 1. Parametry
# Parametry geometryczne
R_in = 0.1  # promień wewnętrznej sfery
R_out = 1.0  # promień zewnętrznej sfery
V_inner = 0
N_theta = 500                      # liczba punktów theta
L_max = 100                # maksymalny stopień l
theta = np.linspace(0, np.pi, N_theta)  # siatka θ ∈ [0, π]
cos_theta = np.cos(theta)
w = np.sin(theta) * (np.pi / (N_theta - 1))  # wagi trapezów z sinθ

# 2. Zdefiniuj funkcję F(θ)
# --- PRZYKŁAD 1: losowe paski
# F = np.where((theta > np.pi/4) & (theta < np.pi/2), 1, -1)

# --- PRZYKŁAD 2: sinusoidalne paski
# F = np.sin(4 * theta)

# --- PRZYKŁAD 3: kombinacja
F = 0.5 * np.sin(3 * theta) + 0.5 * np.where((theta > np.pi/3) & (theta < 2*np.pi/3), 1, -1)

# 3. Oblicz współczynniki C_l
C = np.zeros(L_max + 1)
for l in range(L_max + 1):
    P_l = legendre(l)(cos_theta)
    C[l] = (2 * l + 1) / 2 * np.sum(F * P_l * w)

# 4. Rekonstrukcja funkcji z szeregów
F_reconstructed = np.zeros_like(theta)
for l in range(L_max + 1):
    P_l = legendre(l)(cos_theta)
    F_reconstructed += C[l] * P_l

# 5. Wykresy
plt.figure(figsize=(10, 5))
plt.plot(theta * 180 / np.pi, F, label='Oryginalna funkcja F(θ)', lw=2)
plt.plot(theta * 180 / np.pi, F_reconstructed, label=f'Rekonstrukcja z L_max={L_max}', lw=2, linestyle='--')
plt.xlabel('θ (stopnie)')
plt.ylabel('F(θ)')
plt.title('Rozwój funkcji F(θ) w wielomiany Legendre’a')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Wyznaczanie A_l i B_l przy Φ(R_out, θ) = C_l * P_l, Φ(R_in, θ) = 0
A = np.zeros(L_max + 1)
B = np.zeros(L_max + 1)

for l in range(L_max + 1):
    M = np.array([
        [R_out**l, R_out**(-l - 1)],
        [R_in**l, R_in**(-l - 1)]
    ])
    
    if l == 0:
        rhs = np.array([C[l], V_inner])  # tylko zerowy składnik ma wpływ
    else:
        rhs = np.array([C[l], 0.0])      # reszta = 0
    A[l], B[l] = np.linalg.solve(M, rhs)



@njit
def compute_phi_grid(r_vals, cos_theta, P_all, A, B, L_max):
    Nr = len(r_vals)
    Nt = len(cos_theta)
    result = np.zeros((Nt, Nr))
    
    for l in range(L_max + 1):
        for i in range(Nt):
            for j in range(Nr):
                r = r_vals[j]
                result[i, j] += (A[l] * r**l + B[l] * r**(-l - 1)) * P_all[l, i]
    return result



# --- Przykład: wizualizacja Φ(r, θ) jako heatmapa ---
N_r = 200
r_vals = np.linspace(R_in, R_out, N_r)
R, T = np.meshgrid(r_vals, theta)
PHI = np.zeros_like(R)

# theta, cos_theta wcześniej zdefiniowane
P_all = np.zeros((L_max + 1, N_theta))  # shape: (L_max+1, len(theta))
for l in range(L_max + 1):
    P_all[l, :] = legendre(l)(cos_theta)

V = compute_phi_grid(r_vals, cos_theta, P_all, A, B, L_max)

# Zamień na współrzędne kartezjańskie do wykresu 2D (r, θ)
X = R * np.sin(T)
Y = R * np.cos(T)

plt.figure(figsize=(8, 6))
plt.pcolormesh(X, Y, V, shading='auto', cmap='viridis')
plt.colorbar(label='Φ(r, θ)')
plt.title('Rozwiązanie równania Laplace’a między sferami')
plt.xlabel('x')
plt.ylabel('z')
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()


