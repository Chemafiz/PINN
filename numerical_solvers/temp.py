import numpy as np
from numba import njit, prange
import time
from collections import namedtuple
from functools import partial
import matplotlib.pyplot as plt
import os



def apply_boundary_conditions_constant_value(N, nr, ntheta_max, ntheta_min, vmin=0.0, vmax=1.0):
    step = int(np.floor((ntheta_max - ntheta_min) / (nr - 1)))

    laplace_solutions = np.full((N, nr, ntheta_max), np.nan, dtype=np.float32)

    valid_len = ntheta_max
    for r in range(nr - 1, -1, -1):
        laplace_solutions[:, r, :valid_len] = 1 
        valid_len -= step

    
    boundary_conditions = np.random.uniform(vmin, vmax, size=(N, 1)) 
    laplace_solutions[:, -1, :] = boundary_conditions 

    return laplace_solutions, boundary_conditions


def apply_boundary_conditions_sinusoidal(N, nr, ntheta_max, ntheta_min, amin=0.5, amax=1.0, fmin=1.0, fmax=4.0):
    step = int(np.floor((ntheta_max - ntheta_min) / (nr - 1)))

    laplace_solutions = np.full((N, nr, ntheta_max), np.nan, dtype=np.float32)
    boundary_conditions = np.zeros((N, 2))

    valid_len = ntheta_max
    for r in range(nr - 1, -1, -1):
        laplace_solutions[:, r, :valid_len] = 1
        valid_len -= step

    theta = np.linspace(0, 2 * np.pi, ntheta_max, endpoint=False)
    amplitudes = np.random.uniform(amin, amax, size=(N, 1))  # shape (N, 1)
    frequencies = np.random.randint(fmin, fmax + 1, size=(N, 1))

    # Generujemy sinusoidy (broadcasting przez oś kąta)
    sinusoidal_boundaries = amplitudes * np.sin(frequencies * theta)  
    
    laplace_solutions[:, -1, :] = sinusoidal_boundaries
    boundary_conditions = np.concatenate([amplitudes, frequencies], axis=1) 

    return laplace_solutions, boundary_conditions


@njit(parallel=True)
def generate_data(start_idx, chunk_size, laplace_solutions, r, dr, dtheta):
    for i in prange(chunk_size):
        idx = start_idx + i
        theta_lenghts = np.count_nonzero(~np.isnan(laplace_solutions[idx]), axis=1)
        laplace_solutions[idx] = laplace_solver_sor(laplace_solutions[idx, :, :], r, dr, theta_lenghts=theta_lenghts, maxiter=100000, tol=1e-6, )

@njit
def downsample(data, target_len):
    original_len = data.shape[0]
    factor = original_len // target_len
    downsampled = np.empty(target_len, dtype=np.float32)

    for i in range(target_len):
        start = i * factor
        count = factor
        temp = np.empty(count, dtype=np.float32)
        for j in range(count):
            temp[j] = data[(start + j) % original_len]
        downsampled[i] = np.mean(temp)

    return downsampled


@njit
def upsample(data, target_len):
    original_len = data.shape[0]
    scale = original_len / target_len
    upsampled = np.zeros(target_len, dtype=np.float32)

    # print(data)
    for i in range(target_len):
        pos = i * scale
        idx_low = int(np.floor(pos)) % original_len
        idx_high = (idx_low + 1) % original_len

        weight_high = pos - np.floor(pos)
        weight_low = 1.0 - weight_high

        upsampled[i] = data[idx_low] * weight_low + data[idx_high] * weight_high

    return upsampled



@njit(fastmath=True)
def laplace_solver_sor(u, r, dr, theta_lenghts, omega=1.0, maxiter=100000, tol=1e-6):
    Nr, Ntheta = u.shape

    # theta_lenghts = count_nans_per_row(u)
    coef_r = (1 + dr / (2 * r))
    coef_l = (1 - dr / (2 * r))

    dtheta = 2 * np.pi / theta_lenghts
    coef_theta = (dr**2) / (r**2 * dtheta**2)
    denom = 2 + 2 * coef_theta

    # print(theta_lenghts)
    # print(coef_r)
    # print(coef_l)

    for iteration in range(maxiter):
        max_diff = 0.0

        for i in range(1, Nr-1):
            ri = r[i]

            theta_len = theta_lenghts[i]
            downsampled = downsample(u[i+1, :theta_lenghts[i+1]], theta_len)
            upsampled = upsample(u[i-1, :theta_lenghts[i-1]], theta_len)
            
   

            for j in range(theta_len):
                jp = (j + 1) % theta_len
                jm = (j - 1) % theta_len

                u_new = (coef_r[i] * downsampled[j] + coef_l[i] * upsampled[j] + coef_theta[i] * (u[i, jp] + u[i, jm])) / denom[i]
                diff = u_new - u[i, j]
                u[i, j] += omega * diff
                # u[i, j] = u_new


                if np.abs(diff) > max_diff:
                    max_diff = np.abs(diff)

        # # Obsługa środka r=0 osobno
        # for j in range(Ntheta):
        #     theta_len = theta_lenghts[j]
        #     jp = (j + 1) % theta_len
        #     jm = (j - 1) % theta_len

        #     u_new = 0.25 * (u[1, j] + u[0, jp] + u[0, jm] + u[1, jp])
        #     diff = u_new - u[0, j]
        #     u[0, j] += omega * diff

        #     if np.abs(diff) > max_diff:
        #         max_diff = np.abs(diff)

        if iteration % 100 == 0:
            if max_diff <= tol:
                break

    return u

def visualize_solution(laplace_solution, r_min, r_max):
    nr, ntheta = laplace_solution.shape  # N - liczba próbek, nr - liczba promieni, max_len - liczba wartości theta w każdym promieniu
    
    r_edges = np.linspace(r_min, r_max, nr)  # Krawędzie dla r

    x, y, z = [], [], []
    for r_id, r in enumerate(r_edges):
        theta_edges = np.linspace(0, 2 * np.pi, sum(~np.isnan(laplace_solution[r_id, :])), endpoint=False)
        x.extend(list(r * np.cos(theta_edges)))
        y.extend(list(r * np.sin(theta_edges)))
        z.extend(list(laplace_solution[r_id, :][~np.isnan(laplace_solution[r_id, :])]))
        
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, c=z, s=1)
    plt.colorbar(label='u(r, θ)')
    plt.title('Rozwiązanie równania Laplace\'a w kole')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.show()



if __name__ == "__main__":
    N = 1
    chunk_size = 1
    nr = 100
    ntheta = 100
    ntheta_max=500
    ntheta_min=1
    r_min, r_max = 0, 10

    r = np.linspace(r_min, r_max, nr)
    theta = np.linspace(0, 2*np.pi, ntheta)
    dr = r[1] - r[0]
    dtheta = theta[1] - theta[0]
    # dr = (r_max - r_min) / (nr - 1)
    # dtheta = 2 * np.pi / ntheta

    # laplace_solutions, boundary_conditions = apply_boundary_conditions_constant_value(N, nr, ntheta_max=ntheta_max, ntheta_min=ntheta_min, vmin=0.0, vmax=1.0)
    laplace_solutions, boundary_conditions = apply_boundary_conditions_sinusoidal(N, nr, ntheta_max=ntheta_max, ntheta_min=ntheta_min, amin=1, amax=10.0, fmin=1.0, fmax=10.0)
    # laplace_solutions, boundary_conditions =  apply_boundary_conditions_sinusoidal(N, nr, ntheta, amp_min=1.0, amp_max=100.0, freq_min=1, freq_max=10)

    for start_idx in range(0, N, chunk_size):
        start_time = time.time()
        generate_data(start_idx, chunk_size, laplace_solutions, r, dr, dtheta)
        elapsed = time.time() - start_time
        print(f"Generated {start_idx + chunk_size}/{N}  |  Last chunk time: {elapsed:.2f}s")

    # print(np.count_nonzero(~np.isnan(laplace_solutions[0]), axis=1))
    for i in range(4):
        visualize_solution(laplace_solution=laplace_solutions[i], r_min=0, r_max=10)

    # save_path = 'solutions/polar/'
    # os.makedirs(save_path, exist_ok=True)
    # laplace_solutions = laplace_solutions.astype(np.float32)
    # boundary_conditions = boundary_conditions.astype(np.float32)
    # np.savez_compressed(save_path + 'solutions.npz', laplace_solutions=laplace_solutions, boundary_conditions=boundary_conditions)



