import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt


@njit
def laplace_solver(u, r, dr, dtheta, maxiter=100000, tol=1e-6):
    Nr, Ntheta = u.shape
    ip = np.arange(1, Nr-1) + 1
    im = np.arange(1, Nr-1) - 1
    jp = (np.arange(Ntheta) + 1) % Ntheta
    jm = (np.arange(Ntheta) - 1) % Ntheta

    coef_r = 1 + dr / (2 * r[1:Nr-1])
    coef_l = 1 - dr / (2 * r[1:Nr-1])
    coef_theta = (dr ** 2) / (r[1:Nr-1] ** 2 * dtheta ** 2)
    denom = 2 + 2 * coef_theta
    u_new = u.copy()

    for iteration in range(maxiter):
        u_new[1:Nr-1, :] = (1 / denom[:, None]) * (
            coef_r[:, None] * u[ip, :] +
            coef_l[:, None] * u[im, :] +
            coef_theta[:, None] * (u[1:Nr-1, jp] + u[1:Nr-1, jm])
        )

        u_new[0, :] = 0.25 * (u[1, :] + u[0, jp] + u[0, jm] + u[1, jp])

        if iteration % 100 == 0:
            if np.max(np.abs(u - u_new)) <= tol:
                break

        u, u_new = u_new, u

    return u



if __name__ == "__main__":
    # Parametry siatki
    Nr = 200         # liczba punktów w r
    Ntheta = 200     # liczba punktów w theta (pełny obrót)
    r_min = 0.0
    r_max = 5.0
    dr = (r_max - r_min) / (Nr - 1)
    dtheta = 2 * np.pi / Ntheta

    # Tworzenie siatki
    u = np.zeros((Nr, Ntheta)) # pole u
    r = np.linspace(r_min, r_max, Nr)  # siatka r

    # Warunki brzegowe
    u[-1, :] = 10.0  # Na brzegu koła (r=10) u = 10

    u = laplace_solver(u, r, dr, dtheta, maxiter=100000, tol=1e-6)

    # Zakładając, że masz już r, Ntheta, u, r_min, r_max, Nr zdefiniowane
    r_edges = np.linspace(r_min, r_max, Nr + 1)  # Krawędzie dla r
    theta_edges = np.linspace(0, 2 * np.pi, Ntheta + 1)  # Krawędzie dla theta

    # Stworzenie siatki dla współrzędnych r, theta
    R, Theta = np.meshgrid(r_edges, theta_edges)

    # Przekształcenie do współrzędnych kartezjańskich
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)

    # Tworzenie wykresu
    plt.figure(figsize=(8,6))
    plt.pcolormesh(X, Y, u.T, shading='auto', cmap='inferno')  # Teraz z krawędziami
    plt.colorbar(label='u(r, θ)')
    plt.title('Rozwiązanie równania Laplace\'a w kole o promieniu 10')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.show()