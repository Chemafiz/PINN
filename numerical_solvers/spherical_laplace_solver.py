import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.special import eval_legendre
from tqdm import tqdm
import boundary_conditions as bc


def compute_harmonics_coefficients(theta, V_out, L_max):
    cos_theta = np.cos(theta)
    w = np.sin(theta) * (np.pi / (len(theta) - 1)) 
    C = np.zeros(L_max)

    for l in range(L_max):
        P_l = eval_legendre(l, cos_theta)
        C[l] = (2 * l + 1) / 2 * np.sum(V_out * P_l * w)

    return C


def reconstruct_harmonics(theta, C, L_max):
    cos_theta = np.cos(theta)
    V_out_reconstructed = np.zeros_like(theta)

    for l in range(L_max):
        P_l = eval_legendre(l, cos_theta)
        V_out_reconstructed += C[l] * P_l

    return V_out_reconstructed

def visualize_harmonics(theta, L_max, V, V_reconstructed):
    plt.figure(figsize=(10, 5))
    plt.plot(theta * 180 / np.pi, V, label='Oryginalna funkcja V_out', lw=2)
    plt.plot(theta * 180 / np.pi, V_reconstructed, label=f'Rekonstrukcja z L_max={L_max}', lw=2, linestyle='--')
    plt.xlabel('theta θ')
    plt.ylabel('V_out(θ)')
    plt.title('Wizualizacja harmonik sferycznych')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compute_radial_coefficients(V_in, C, R_in, R_out, L_max):
    A = np.zeros(L_max)
    B = np.zeros(L_max)

    for l in range(L_max):
        M = np.array([
            [R_out**l, R_out**(-l - 1)],
            [R_in**l, R_in**(-l - 1)]
        ])
        
        if l == 0:
            rhs = np.array([C[l], V_in])  
        else:
            rhs = np.array([C[l], 0.0])     
        A[l], B[l] = np.linalg.solve(M, rhs)

    return A, B


def pre_compute_P_coefficients(theta, L_max):
    cos_theta = np.cos(theta)
    P_l = np.zeros((L_max, len(theta)))  
    for l in range(L_max):
        P_l[l, :] = eval_legendre(l, cos_theta)
    return P_l


@njit
def compute_laplace_solution(r, theta, P_l, A, B, L_max):
    Nr = len(r)
    Nt = len(theta)

    result = np.zeros((Nt, Nr))
    
    for l in range(L_max):
        for i in range(Nt):
            for j in range(Nr):
                rj = r[j]
                result[i, j] += (A[l] * rj**l + B[l] * rj**(-l - 1)) * P_l[l, i]
    return result


def visualize_laplace_solution(r, theta, V_half):
    R, T = np.meshgrid(r, theta)
    X = R * np.sin(T)
    Y = R * np.cos(T)

    X_full = np.concatenate((-X[::-1, :], X), axis=0)
    Y_full = np.concatenate((Y[::-1, :], Y), axis=0)
    V_full = np.concatenate((V_half[::-1, :], V_half), axis=0)

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X_full, Y_full, V_full, shading='nearest', cmap='viridis')
    plt.colorbar(label='Φ(r, θ)')
    plt.title('Rozwiązanie równania Laplace’a')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.show()


def generate_train_samples(
        N=1000, 
        R_in_ranges=(0.1, 0.7),
        R_out=1.0,
        V_in_ranges=(-10, 10),
        N_theta=100,
        L_max=100,
        boundary_conditions=None,
        path="solutions.npz"
        ):
    

    X_data = []
    Y_data = []
    
    for _ in tqdm(range(N)):
        R_in = np.random.uniform(*R_in_ranges)
        V_in = np.random.uniform(*V_in_ranges)
        theta = np.linspace(0, np.pi, N_theta)
        
        bc_func = np.random.choice(boundary_conditions)
        V_out = bc_func(theta)
        
        C = compute_harmonics_coefficients(theta, V_out, L_max)
        
        A, B = compute_radial_coefficients(V_in, C, R_in, R_out, L_max)
        
        X = np.hstack(([R_in, R_out, V_in], C))
        Y = np.hstack((A, B))
        
        X_data.append(X)
        Y_data.append(Y)


    np.savez_compressed(path, X=np.array(X_data), Y=np.array(Y_data))
    print(f"Saved to {path}!!!")




if __name__ == "__main__":

    R_in = 0.1  # promień wewnętrznej sfery
    R_out = 1.0  # promień zewnętrznej sfery
    N_theta, N_r = 100, 100 
    L_max = 100                # maksymalny stopień l
    theta = np.linspace(0, np.pi, N_theta) 

    V_out = bc.v_out_mixed(theta)
    V_in = 0

    C = compute_harmonics_coefficients(theta, V_out, L_max)

    V_out_reconstructed = reconstruct_harmonics(theta, C, L_max)
    visualize_harmonics(theta, L_max, V_out, V_out_reconstructed)

    A, B = compute_radial_coefficients(V_in, C, R_in, R_out, L_max)

    r = np.linspace(R_in, R_out, N_r)
    P_l = pre_compute_P_coefficients(theta, L_max)
    V = compute_laplace_solution(r, theta, P_l, A, B, L_max)

    visualize_laplace_solution(r, theta, V)


    generate_train_samples(
        N=2000, 
        R_in_ranges=(0.5, 0.5),
        R_out=1.0,
        V_in_ranges=(-1, 1),
        N_theta=200,
        L_max=100,
        boundary_conditions=[bc.v_out_step, bc.v_out_sin, bc.v_out_mixed],
        path = "/home/ml_master/projects/PINN/solutions/sphere_symmetrical/solutions.npz"
        )

    


