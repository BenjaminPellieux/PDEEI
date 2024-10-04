import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.linalg import solve_banded

# Paramètres physiques
L = 1  # Dimension = du mur (m)
l = 1
h = 1
k = 1 #80  Conductivité thermique du béton (W/m·K)
T_left = 100  # Température à la frontière gauche (°C) - Dirichlet
T_right_Dirichlet = 25  # Température à la frontière droite (°C) - Dirichlet
N = 1000  # Nombre de volumes finis
dx = L / N  # Taille des volumes finis
S = h * l  # Surface
src = 100 # source de challeur au centre 
v = dx * S # Volume
p = 2200 #7800 # kg / m3
c =  880 #450  (J K−1 kg−1) 
t_total = 3600 * 24
ephoc = int(t_total *  0.1)
dt = 1


# Initialisation des températures
T_old = np.ones(N) * 20  # Température initiale
T_new = np.copy(T_old)

def init_tridiagonal_matrix() -> (np.ndarray, np.ndarray):
    """ Initialisation de la matrice tridiagonale et du vecteur B """
    lower_diag = np.ones(N-1) * -(k * S) / dx
    main_diag = np.ones(N) * (2 * (k * S) / dx + ((p * c * v) / dt))
    upper_diag = np.ones(N-1) * -(k * S) / dx
    B = np.copy(T_old)

    # Conditions aux limites Dirichlet à gauche
    main_diag[0] = (k * S) / (dx / 2) + (k * S) / dx
    B[0] = (k * S) / (dx / 2) * T_left + T_old[0]

    return lower_diag, main_diag, upper_diag, B

def solve_dirichlet() -> np.ndarray:
    """ Résolution du problème avec conditions de Dirichlet-Dirichlet """
    lower_diag, main_diag, upper_diag, B = init_tridiagonal_matrix()

    # Condition Dirichlet à droite
    main_diag[-1] = (k * S) / dx + (k * S) / (dx / 2) + ((p * c * v) / dt)
    B[-1] = (k * S) / (dx / 2) * T_right_Dirichlet + T_old[-1]

    # Résolution via solve_banded pour matrice tridiagonale
    ab = np.zeros((3, N))
    ab[0, 1:] = upper_diag  # Surdiagonale
    ab[1, :] = main_diag    # Diagonale principale
    ab[2, :-1] = lower_diag  # Sous-diagonale

    T_new = solve_banded((1, 1), ab, B)
    return T_new

def solve_dirichlet_neumann() -> np.ndarray:
    """ Résolution du problème avec conditions Dirichlet-Neumann """
    lower_diag, main_diag, upper_diag, B = init_tridiagonal_matrix()

    # Condition Neumann à droite
    main_diag[-1] = (k * S) / dx + ((p * c * v) / dt)
    B[-1] = T_old[-1]

    # Résolution via solve_banded pour matrice tridiagonale
    ab = np.zeros((3, N))
    ab[0, 1:] = upper_diag  # Surdiagonale
    ab[1, :] = main_diag    # Diagonale principale
    ab[2, :-1] = lower_diag  # Sous-diagonale

    T_new = solve_banded((1, 1), ab, B)
    return T_new

# Résolution des deux scénarios
x = np.linspace(0, L, N)
choix = int(input("Entrez choix: \n1: Dirichlet-Dirichlet\n2: Dirichlet-Neumann\n "))

for i in range(0, t_total, dt):
    for j in range(N):
        T_old[j] = T_new[j] * ((c * p * v) / dt)

    if choix == 1:
        T_new = solve_dirichlet()
    else:
        T_new = solve_dirichlet_neumann()

    # Affichage des courbes à des intervalles réguliers, par ex. toutes les 600 secondes
    if i % ephoc == 0:
        plt.plot(x, T_new, label=f'Temps = {i} s')

# Affichage final des résultats
plt.title('Distribution de la température dans le mur (régime transitoire)')
plt.xlabel('Position (m)')
plt.ylabel('Température (°C)')
plt.legend()
plt.grid(True)
plt.show()
