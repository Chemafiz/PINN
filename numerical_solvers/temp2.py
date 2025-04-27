import numpy as np
import matplotlib.pyplot as plt

# Parametry siatki
Nr = 20         # liczba punktów w r
Ntheta = 20     # liczba punktów w theta (pełny obrót)
r_min = 0.0
r_max = 5.0
dr = (r_max - r_min) / (Nr - 1)
dtheta = 2 * np.pi / Ntheta

# Tworzenie siatki
u = np.zeros((Nr, Ntheta)) # pole u
r = np.linspace(r_min, r_max, Nr)  # siatka r

# Warunki brzegowe
u[-1, :] = 10.0  # Na brzegu koła (r=10) u = 10

# Iteracyjna relaksacja
for iteration in range(2000):
    u_new = np.copy(u)
    for i in range(1, Nr-1):         # pomijamy i=0 (środek) i i=Nr-1 (brzeg, bo tam mamy warunek brzegowy)
        for j in range(Ntheta):
            ip = i + 1
            im = i - 1
            jp = (j + 1) % Ntheta    # cykliczne po theta
            jm = (j - 1) % Ntheta

           
            coef_r = (1 + dr / (2 * r[i]))
            coef_l = (1 - dr / (2 * r[i]))
            coef_theta = (dr ** 2) / (r[i] ** 2 * dtheta ** 2)
            denom = 2 + 2 * coef_theta

            u_new[i, j] = (1 / denom) * (
                coef_r * u[ip, j] +
                coef_l * u[im, j] +
                coef_theta * (u[i, jp] + u[i, jm])
            )

            u_new[0, j] = 0.25 * (u[1, j] + u[0, jp] + u[0, jm] + u[1, jp])

        
    if iteration % 100 == 0:
            max_diff = np.max(np.abs(u - u_new))
            print(f"error = {max_diff}")

    u = u_new

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