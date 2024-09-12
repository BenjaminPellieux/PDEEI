import numpy as np
import matplotlib.pyplot as plt

# Paramètres physiques
L = 1  # Longueur du mur (m)
k = 1.5  # Conductivité thermique du béton (W/m·K)
T_left = 100  # Température à la frontière gauche (°C) - Condition de Dirichlet
T_right = 20  # Température à la frontière droite (°C) - Condition de Dirichlet
N = 1000  # Nombre de volumes finis
dx = L / N  # Taille des volumes finis

# Construction de la matrice du système
A = np.zeros((N, N))
b = np.zeros(N)

# Remplir la matrice A et le vecteur b
for i in range(1, N-1):
    A[i, i-1] = k / dx**2
    A[i, i] = -2 * k / dx**2
    A[i, i+1] = k / dx**2

# Conditions aux limites (Dirichlet)
A[0, 0] = 1  # Condition de Dirichlet à gauche
b[0] = T_left

A[-1, -1] = 1  # Condition de Dirichlet à droite
b[-1] = T_right

# Résoudre le système linéaire Ax = b
T = np.linalg.solve(A, b)

# Affichage des résultats
x = np.linspace(0, L, N)
plt.plot(x, T, marker='o')
plt.title('Distribution de la température dans le mur')
plt.xlabel('Position (m)')
plt.ylabel('Température (°C)')
plt.grid(True)
plt.show()

