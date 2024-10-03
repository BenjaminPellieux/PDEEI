import numpy as np
import matplotlib.pyplot as plt
# https://fr.wikipedia.org/wiki/Liste_de_conductivit%C3%A9s_thermiques
# Paramètres physiques
L: int = 1  # Dimension = du mur (m)
l: int = 1
h: int = 1
k: float = 0.8  # Conductivité thermique du béton (W/m·K)
T_left: int = 100  # Température à la frontière gauche (°C) - Dirichlet
T_right_Dirichlet: int = 25  # Température à la frontière droite (°C) - Dirichlet
N: int = 1000  # Nombre de volumes finis
dx: float = L / N  # Taille des volumes finis
S: int = h * l 



def init_matrix() -> [np.ndarray, np.ndarray]: 
    """ Construction de la matrice du système """
    A, B = np.zeros((N, N)), np.zeros(N)
    for i in range(1, N-1):
        A[i, i - 1] = -(k * S) / dx
        A[i, i + 1] = -(k * S) / dx
        A[i, i] = 2 * (k * S) / dx
    
    # Condition de Dirichlet à gauche
    A[0, 0] = (k * S) / (dx / 2) + (k * S) / dx  # calcul de la condition de Dirichlet à gauche
    A[0, 1] = -(k * S) / dx
    B[0] = (k * S) / (dx / 2) * T_left

    return A, B
def solve_dirichlet() -> np.ndarray:
    
    """ Fonction pour résoudre le problème avec conditions 
    de Dirichlet aux deux extrémités"""
    A, b = init_matrix()

    A[N - 1, N - 2] = -(k * S) / dx 
    A[N - 1, N - 1] = (k * S) / dx + (k * S) / (dx / 2)
    b[N - 1] = (k * S) / (dx / 2) * T_right_Dirichlet
    
    T: np.ndarray = np.linalg.solve(A, b)
    return T

def solve_dirichlet_neumann() -> np.ndarray:
    """ Fonction pour résoudre le problème avec condition 
    de Dirichlet à gauche et Neumann à droite """
    A, b  = init_matrix()
    
    A[N - 1, N - 2] = -(k * S) / dx  
    A[N - 1, N - 1] = (k * S) / dx
    b[N - 1] = 0
    
    T: np.ndarray = np.linalg.solve(A, b)
    
    return T


# Résolution des deux scénarios
T_dirichlet: np.ndarray = solve_dirichlet()
T_dirichlet_neumann: np.ndarray = solve_dirichlet_neumann()
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