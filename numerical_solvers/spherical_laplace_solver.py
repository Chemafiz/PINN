import numpy as np
from numba import njit, prange
import time
from collections import namedtuple
from functools import partial
import matplotlib.pyplot as plt
import os



def apply_boundary_conditions(N, nr, ntheta, vmin=0.0, vmax=1.0):
    laplace_solutions = np.zeros((N, nr, ntheta))
    
    boundary_conditions = np.random.uniform(vmin, vmax, size=(N, 1)) 
    laplace_solutions[:, -1, :] = boundary_conditions 

    return laplace_solutions, boundary_conditions


@njit(parallel=True)
def generate_data(start_idx, chunk_size, laplace_solutions, r, dr, dtheta):
    for i in prange(chunk_size):
        idx = start_idx + i
        laplace_solutions[idx] = laplace_solver_sor(laplace_solutions[idx], r, dr, dtheta, maxiter=100000, tol=1e-6)


@njit(fastmath=True)
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

    coef_r = coef_r[:, None]
    coef_l = coef_l[:, None]
    coef_theta = coef_theta[:, None]
    denom = 1 / denom[:, None]

    u_new = u.copy()

    for iteration in range(maxiter):
        u_new[1:Nr-1, :] = denom * (
            coef_r * u[ip, :] +
            coef_l * u[im, :] +
            coef_theta * (u[1:Nr-1, jp] + u[1:Nr-1, jm])
        )

        # u_new[0, :] = 0.25 * (u[1, :] + u[0, jp] + u[0, jm] + u[1, jp])
        u_new[0, :] = u[1, :]

        if iteration % 100 == 0:
            if np.max(np.abs(u - u_new)) <= tol:
                break

        u, u_new = u_new, u

    return u


@njit(fastmath=True)
def laplace_solver_sor(u, r, dr, dtheta, omega=1.9, maxiter=100000, tol=1e-6):
    Nr, Ntheta = u.shape

    dr2 = dr * dr
    dtheta2 = dtheta * dtheta

    for iteration in range(maxiter):
        max_diff = 0.0

        for i in range(1, Nr-1):
            ri = r[i]
            rip = r[i+1]
            rim = r[i-1]

            coef_r = (1 + dr / (2 * ri))
            coef_l = (1 - dr / (2 * ri))
            coef_theta = (dr2) / (ri * ri * dtheta2)
            denom = 2 + 2 * coef_theta

            for j in range(Ntheta):
                jp = (j + 1) % Ntheta
                jm = (j - 1) % Ntheta

                u_new = (coef_r * u[i+1, j] + coef_l * u[i-1, j] + coef_theta * (u[i, jp] + u[i, jm])) / denom
                diff = u_new - u[i, j]
                u[i, j] += omega * diff

                if np.abs(diff) > max_diff:
                    max_diff = np.abs(diff)

        # Obsługa środka r=0 osobno
        for j in range(Ntheta):
            jp = (j + 1) % Ntheta
            jm = (j - 1) % Ntheta

            u_new = 0.25 * (u[1, j] + u[0, jp] + u[0, jm] + u[1, jp])
            diff = u_new - u[0, j]
            u[0, j] += omega * diff

            if np.abs(diff) > max_diff:
                max_diff = np.abs(diff)

        if iteration % 100 == 0:
            if max_diff <= tol:
                break

    return u


def visualize_solution(laplace_solution):
    nr, ntheta = laplace_solution.shape

    r_edges = np.linspace(r_min, r_max, nr + 1)  # Krawędzie dla r
    theta_edges = np.linspace(0, 2 * np.pi, ntheta + 1)  # Krawędzie dla theta

    # Stworzenie siatki dla współrzędnych r, theta
    R, Theta = np.meshgrid(r_edges, theta_edges)

    # Przekształcenie do współrzędnych kartezjańskich
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)

    # Tworzenie wykresu
    plt.figure(figsize=(8,6))
    plt.pcolormesh(X, Y, laplace_solution.T, shading='auto', cmap='inferno')  # Teraz z krawędziami
    plt.colorbar(label='u(r, θ)')
    plt.title('Rozwiązanie równania Laplace\'a w kole o promieniu 10')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    N = 1000
    chunk_size = 100
    nr = 100
    ntheta = 100
    r_min, r_max = 0, 10

    r = np.linspace(r_min, r_max, nr)
    theta = np.linspace(0, 2*np.pi, ntheta)
    # dr = r[1] - r[0]
    # dtheta = theta[1] - theta[0]
    dr = (r_max - r_min) / (nr - 1)
    dtheta = 2 * np.pi / ntheta

    laplace_solutions, boundary_conditions = apply_boundary_conditions(N, nr, ntheta, vmin=-10.0, vmax=10.0)

    for start_idx in range(0, N, chunk_size):
        start_time = time.time()
        generate_data(start_idx, chunk_size, laplace_solutions, r, dr, dtheta)
        elapsed = time.time() - start_time
        print(f"Generated {start_idx + chunk_size}/{N}  |  Last chunk time: {elapsed:.2f}s")


    
    # for i in range(4):
    #     visualize_solution(laplace_solution=laplace_solutions[i])

    save_path = 'solutions/polar/'
    os.makedirs(save_path, exist_ok=True)
    laplace_solutions = laplace_solutions.astype(np.float32)
    boundary_conditions = boundary_conditions.astype(np.float32)
    np.savez_compressed(save_path + 'solutions.npz', laplace_solutions=laplace_solutions, boundary_conditions=boundary_conditions)



