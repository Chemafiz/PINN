import numpy as np
from numba import njit, prange
import time
from collections import namedtuple
from functools import partial
import matplotlib.pyplot as plt
import os



def apply_boundary_conditions_constant_value(N, nr, ntheta, vmin=0.0, vmax=1.0):
    laplace_solutions = np.zeros((N, nr, ntheta))
    
    boundary_conditions = np.random.uniform(vmin, vmax, size=(N, 1)) 
    laplace_solutions[:, -1, :] = boundary_conditions 

    return laplace_solutions, boundary_conditions


def apply_boundary_conditions_sinusoidal(
                                        N, nr, ntheta,
                                        amp_min=1.0, amp_max=10.0,
                                        freq_min=1, freq_max=10,
                                        bias_min=0.0, bias_max=0.0
                                        ):
    
    laplace_solutions = np.zeros((N, nr, ntheta))
    boundary_conditions = np.zeros((N, 3))  # amplitude, frequency, bias

    amplitudes = np.random.uniform(amp_min, amp_max, size=N)
    frequencies = np.random.randint(freq_min, freq_max + 1, size=N)
    biases = np.random.uniform(bias_min, bias_max, size=N)

    theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)  
    
    theta_matrix = theta[np.newaxis, :]  
    freq_matrix = frequencies[:, np.newaxis]  
    amp_matrix = amplitudes[:, np.newaxis] 
    bias_matrix = biases[:, np.newaxis]    

    boundary_values = amp_matrix * np.sin(freq_matrix * theta_matrix) + bias_matrix 

    laplace_solutions[:, -1, :] = boundary_values
    boundary_conditions[:, 0] = amplitudes
    boundary_conditions[:, 1] = frequencies
    boundary_conditions[:, 2] = biases

    return laplace_solutions, boundary_conditions


def add_central_hole(laplace_solutions, boundary_conditions, r_min=0.05, r_max=0.4):
    N, nr, ntheta = laplace_solutions.shape
    zero_radius = np.random.uniform(r_min, r_max, size=N)
    radius_indices = (zero_radius * (nr - 1)).astype(int)

    for i in range(N):
        laplace_solutions[i, radius_indices[i], :] = 0.0  

    boundary_conditions = np.concatenate([boundary_conditions, zero_radius[:, np.newaxis]], axis=1)

    return laplace_solutions, boundary_conditions



@njit(parallel=True)
def generate_data(start_idx, chunk_size, laplace_solutions, r, dr, dtheta):
    for i in prange(chunk_size):
        idx = start_idx + i
        laplace_solutions[idx] = laplace_solver_sor(laplace_solutions[idx], r, dr, dtheta, maxiter=100000, tol=1e-6)


@njit(parallel=True)
def generate_data_hole(start_idx, chunk_size, laplace_solutions, r, dr, dtheta, zero_circle_indices):
    for i in prange(chunk_size):
        idx = start_idx + i
        z_idx = zero_circle_indices[idx]
        laplace_solutions[idx] = laplace_solver_sor_hole(laplace_solutions[idx], r, dr, dtheta,
                                                     zero_circle_idx=z_idx,
                                                     maxiter=100000, tol=1e-6)

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

        if iteration % 100 == 0:
            if max_diff <= tol:
                break

    return u


@njit(fastmath=True)
def laplace_solver_sor_hole(u, r, dr, dtheta, zero_circle_idx=-1, omega=1.9, maxiter=100000, tol=1e-6):
    Nr, Ntheta = u.shape
    dr2 = dr * dr
    dtheta2 = dtheta * dtheta

    for iteration in range(maxiter):
        max_diff = 0.0

        for i in range(1, Nr - 1):
            ri = r[i]

            coef_r = (1 + dr / (2 * ri))
            coef_l = (1 - dr / (2 * ri))
            coef_theta = (dr2) / (ri * ri * dtheta2)
            denom = 2 + 2 * coef_theta

            for j in range(Ntheta):
                jp = (j + 1) % Ntheta
                jm = (j - 1) % Ntheta

                u_new = (coef_r * u[i + 1, j] +
                         coef_l * u[i - 1, j] +
                         coef_theta * (u[i, jp] + u[i, jm])) / denom

                diff = u_new - u[i, j]
                u[i, j] += omega * diff

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
    plt.pcolormesh(X, Y, laplace_solution.T, shading='auto', cmap='inferno')
    plt.colorbar(label='u(r, θ)')
    plt.title('Rozwiązanie równania Laplace')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    nr, ntheta = 200, 250
    print(f"Leci {nr} oraz {ntheta}")
    N = 1000
    chunk_size = 100
    r_min, r_max = 0, 1

    r = np.linspace(r_min, r_max, nr)
    theta = np.linspace(0, 2*np.pi, ntheta)
    dr = r[1] - r[0]
    dtheta = theta[1] - theta[0]

    # laplace_solutions, boundary_conditions = apply_boundary_conditions_constant_value(N, nr, ntheta, vmin=-100.0, vmax=100.0)
    laplace_solutions, boundary_conditions =  apply_boundary_conditions_sinusoidal(
                                                                                    N, nr, ntheta,
                                                                                    amp_min=1.0, amp_max=10.0,
                                                                                    freq_min=1, freq_max=20,
                                                                                    bias_min=-10.0, bias_max=10
                                                                                    )

    laplace_solutions, boundary_conditions = add_central_hole(
                                                            laplace_solutions,
                                                            boundary_conditions,
                                                            r_min=0.1,
                                                            r_max=0.6
                                                            )
    
    zero_circle_indices = (boundary_conditions[:, 3] * (nr - 1)).astype(np.int32)


    for start_idx in range(0, N, chunk_size):
        start_time = time.time()
        # generate_data(start_idx, chunk_size, laplace_solutions, r, dr, dtheta)
        generate_data_hole(start_idx, chunk_size, laplace_solutions, r, dr, dtheta, zero_circle_indices)
        elapsed = time.time() - start_time
        print(f"Generated {start_idx + chunk_size}/{N}  |  Last chunk time: {elapsed:.2f}s")


    
    # for i in range(10):
    #     visualize_solution(laplace_solution=laplace_solutions[i])


    save_path = "/home/ml_master/projects/PINN/solutions/const_with_hole/"
    os.makedirs(save_path, exist_ok=True)
    laplace_solutions = laplace_solutions.astype(np.float32)
    boundary_conditions = boundary_conditions.astype(np.float32)
    np.savez_compressed(save_path + f'solutions_{nr}_{ntheta}.npz', laplace_solutions=laplace_solutions, boundary_conditions=boundary_conditions)



