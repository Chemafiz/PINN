import numpy as np
from numba import njit, prange
import time
from collections import namedtuple
from functools import partial
import matplotlib.pyplot as plt
import os

@njit(fastmath=True)
def laplace_2d_converge(u, boundaries=None, maxiter=100000, tol=1e-6):
    u_new = u.copy()
    for iteration in range(maxiter):
        u_new[1:-1, 1:-1] = 0.25 * (
            u[2:, 1:-1] +
            u[:-2, 1:-1] +
            u[1:-1, 2:] +
            u[1:-1, :-2]
        )

        if iteration % 10 == 0:
            diff = np.abs(u_new - u).max()
            if diff < tol:
                break

        u, u_new = u_new, u
    return u_new

@njit(fastmath=True)
def laplace_2d_converge_circle(u, boundaries=None, maxiter=100000, tol=1e-6):
    ny, nx = u.shape
    u_new = u.copy()
    center_x, center_y, radius = boundaries[4], boundaries[5], boundaries[6]

    y = np.arange(ny).reshape(ny, 1)
    x = np.arange(nx).reshape(1, nx)
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2

    for iteration in range(maxiter):
        u_new[1:-1, 1:-1] = 0.25 * (
            u[2:, 1:-1] +
            u[:-2, 1:-1] +
            u[1:-1, 2:] +
            u[1:-1, :-2]
        )

        indices = np.where(mask)
        for k in range(indices[0].shape[0]):
            i = indices[0][k]
            j = indices[1][k]
            u_new[i, j] = 0.0

        if iteration % 10 == 0:
            diff = np.abs(u_new - u).max()
            if diff < tol:
                break

        u, u_new = u_new, u
    return u_new


def apply_boundary_conditions(N, dimensions, obj):
    laplace_solutions = np.zeros((N, *dimensions))
    boundaries_ranges =[
        [0.0, 1.0],   # top
        [0.0, 1.0],   # bottom
        [0.0, 1.0],  # left
        [0.0, 1.0]    # right
    ]


    if obj.type == "none":
        boundaries_ranges = np.array(boundaries_ranges)
        low = boundaries_ranges[:, 0]
        high =boundaries_ranges[:, 1]
        boundary_conditions = np.random.uniform(low, high, size=(N, 4))

        laplace_solutions[:, 0, :] = boundary_conditions[:, 0][:, None]   # top
        laplace_solutions[:, -1, :] = boundary_conditions[:, 1][:, None]  # bottom
        laplace_solutions[:, :, 0] = boundary_conditions[:, 2][:, None]   # left
        laplace_solutions[:, :, -1] = boundary_conditions[:, 3][:, None]  # right

        return laplace_solutions, boundary_conditions

    elif obj.type == "circle":
        boundaries_ranges.extend([
            [obj.x_min, obj.x_max], # x center
            [obj.y_min, obj.y_max], # y center
            [obj.radius_min, obj.radius_max], # radius
            ])
        boundaries_ranges = np.array(boundaries_ranges)
        low = boundaries_ranges[:, 0]
        high =boundaries_ranges[:, 1]
        boundary_conditions = np.random.uniform(low, high, size=(N, 7))

        laplace_solutions[:, 0, :] = boundary_conditions[:, 0][:, None]   # top
        laplace_solutions[:, -1, :] = boundary_conditions[:, 1][:, None]  # bottom
        laplace_solutions[:, :, 0] = boundary_conditions[:, 2][:, None]   # left
        laplace_solutions[:, :, -1] = boundary_conditions[:, 3][:, None]  # right

        return laplace_solutions, boundary_conditions



@njit(parallel=True)
def generate_data(start_idx,
                chunk_size,
                laplace_solutions,
                boundary_conditions,
                laplace_solver
                ):
    for i in prange(chunk_size):
        idx = start_idx + i
        laplace_solutions[idx] = laplace_solver(u=laplace_solutions[idx], boundaries=boundary_conditions[idx], maxiter=100000, tol=1e-6)



if __name__ == "__main__":
    dimensions = (100, 100)
    N = 5000
    chunk_size = 5000
    Circle = namedtuple("Circle", ["type", "x_min", "x_max", "y_min", "y_max", "radius_min", "radius_max", "solver"])
    Blank = namedtuple("Blank", ["type", "solver"])

    # obj = Blank(type="none", solver=laplace_2d_converge)
    obj = Circle(type="circle",
                x_min = 30, x_max = 70,
                y_min = 30, y_max = 70,
                radius_min = 5, radius_max = 20,
                solver = laplace_2d_converge_circle
                )



    laplace_solutions, boundary_conditions = apply_boundary_conditions(N, dimensions, obj)

    for start_idx in range(0, N, chunk_size):
        start_time = time.time()
        generate_data(start_idx, chunk_size, laplace_solutions, boundary_conditions, obj.solver)
        elapsed = time.time() - start_time
        print(f"Generated {start_idx + chunk_size}/{N}  |  Last chunk time: {elapsed:.2f}s")


    for i in range(4):
        fig = plt.figure(figsize=(5, 5))
        plt.pcolormesh(laplace_solutions[i], cmap="hot")
        plt.show()


    save_path = 'solutions/circle/'
    os.makedirs(save_path, exist_ok=True)
    laplace_solutions = laplace_solutions.astype(np.float32)
    boundary_conditions = boundary_conditions.astype(np.float32)
    np.savez_compressed(save_path + 'solutions.npz', laplace_solutions=laplace_solutions, boundary_conditions=boundary_conditions)





