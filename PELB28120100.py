import numpy as np
import matplotlib.pyplot as plt

# https://fr.wikipedia.org/wiki/Liste_de_conductivit%C3%A9s_thermiques
# Paramètres physiques
L = 1  # Longueur du mur (m)
k = 0.8  # Conductivité thermique du béton (W/m·K)
T_left = 100  # Température à la frontière gauche (°C) - Dirichlet
T_right_Dirichlet = 25  # Température à la frontière droite (°C) - Dirichlet
q_right_Neumann = 19.85  # Flux thermique à la frontière droite (W/m^2) - Neumann
N = 1000  # Nombre de volumes finis
dx = L / N  # Taille des volumes finis

# Fonction pour résoudre le problème avec conditions de Dirichlet aux deux extrémités
def solve_dirichlet():
    # Construction de la matrice du système
    A = np.zeros((N, N))
    b = np.zeros(N)

    # Remplir la matrice A pour la conduction thermique
    for i in range(1, N-1):
        A[i, i-1] = k / dx**2
        A[i, i] = -2 * k / dx**2
        A[i, i+1] = k / dx**2

    # Conditions aux limites de Dirichlet
    A[0, 0] = 1  # Condition de Dirichlet à gauche
    b[0] = T_left

    A[-1, -1] = 1  # Condition de Dirichlet à droite
    b[-1] = T_right_Dirichlet

    # Résoudre le système linéaire Ax = b
    T = np.linalg.solve(A, b)
    return T

# Fonction pour résoudre le problème avec condition de Dirichlet à gauche et Neumann à droite
def solve_dirichlet_neumann():
    # Construction de la matrice du système
    A = np.zeros((N, N))
    b = np.zeros(N)

    # Remplir la matrice A pour la conduction thermique
    for i in range(1, N-1):
        A[i, i-1] = k / dx**2
        A[i, i] = -2 * k / dx**2
        A[i, i+1] = k / dx**2

    # Condition de Dirichlet à gauche
    A[0, 0] = 1
    b[0] = T_left

    # Condition de Neumann à droite
    A[-1, -1] = -k / dx
    A[-1, -2] = k / dx
    b[-1] = q_right_Neumann

    # Résoudre le système linéaire Ax = b
    T = np.linalg.solve(A, b)
    return T

# Résolution des deux scénarios
T_dirichlet = solve_dirichlet()
T_dirichlet_neumann = solve_dirichlet_neumann()

# Affichage des résultats
x = np.linspace(0, L, N)

plt.plot(x, T_dirichlet, label='Dirichlet')
plt.plot(x, T_dirichlet_neumann, label='Dirichlet-Neumann')
plt.title('Distribution de la température dans le mur')
plt.xlabel('Position (m)')
plt.ylabel('Température (°C)')
plt.legend()
plt.grid(True)
plt.show()
