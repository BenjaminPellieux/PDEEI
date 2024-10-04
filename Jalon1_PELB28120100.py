import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import diags
from scipy.linalg import solve_banded

##########################################
#                Constantes              #
##########################################


Form_liste = ["Dirichlet-Dirichlet","Dirichlet-Neuman"]
# Paramètres physiques
L = 1  # Dimension = du mur (m)
l = 1
h = 1
k = 1  # Conductivité thermique du béton (W/m·K)
T_left = 100  # Température à la frontière gauche (°C) - Dirichlet
T_right_Dirichlet = 25  # Température à la frontière droite (°C) - Dirichlet
N = 1000  # Nombre de volumes finis
dx = L / N  # Taille des volumes finis
S = h * l  # Surface
src = 100  # Source de chaleur au centre
v = dx * S  # Volume
p = 2200  # kg/m3
c = 880  # (J/K·kg)
t_total = 3600 * 72  # Simulation sur 72 heures
ephoc = int(t_total * 0.1)
dt = 1800  # Intervalle de temps en secondes (30 minutes)

# Initialisation des températures
T_old = np.ones(N) * 20  # Température initiale
T_new = np.copy(T_old)

# Stocker les résultats pour afficher en 3D
all_temperatures = []

##########################################
#         Fonctions de simulation        #
##########################################

def init_tridiagonal_matrix() -> (np.ndarray, np.ndarray):
    """ Initialisation de la matrice tridiagonale et du vecteur B """
    lower_diag = np.ones(N - 1) * -(k * S) / dx
    main_diag = np.ones(N) * (2 * (k * S) / dx + ((p * c * v) / dt))
    upper_diag = np.ones(N - 1) * -(k * S) / dx
    B = np.copy(T_old)

    # Conditions aux limites Dirichlet à gauche
    main_diag[0] = (k * S) / (dx / 2) + (k * S) / dx
    B[0] = (k * S) / (dx / 2) * T_left + T_old[0]
    B[N // 2] += src

    return lower_diag, main_diag, upper_diag, B

def solve_dirichlet_dirichlet() -> np.ndarray:
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

    return solve_banded((1, 1), ab, B)

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

    return solve_banded((1, 1), ab, B)

##########################################
#           Boucle de simulation         #
##########################################

# Résolution des deux scénarios
x = np.linspace(0, L, N)
temps = []
choix = int(input("Bonjour bienvenu dans la simulation de Benjamin PELLIEUX.\nEntrez votre choix: \n1: Dirichlet-Dirichlet\n2: Dirichlet-Neumann\n "))
if choix == 1:
    print(f"[INFO] Running Dirichlet-Dirichlet")
else:
    print(f"[INFO] Running Dirichlet-Neumann simunation ")
print(f"[INFO] Ephoc {ephoc}s")

for i in range(0, t_total, dt):
    
    for j in range(N):
        T_old[j] = T_new[j] * ((c * p * v) / dt)

    if choix == 1:
        T_new = solve_dirichlet_dirichlet()
    else:
        T_new = solve_dirichlet_neumann()

    # Stocker les résultats pour l'affichage 3D
    all_temperatures.append(np.copy(T_new))
    temps.append(i)

##########################################
#         Affichage 3D des résultats     #
##########################################

# Convertir les données pour la visualisation
all_temperatures = np.array(all_temperatures)  # (temps, position)
temps = np.array(temps)
X, Y = np.meshgrid(x, temps)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Afficher la température en fonction du temps et de la position
ax.plot_surface(X, Y, all_temperatures, cmap='plasma')

ax.set_title(f'Évolution de la température dans le mur \n avec la condition {Form_liste[choix-1]}')
ax.set_xlabel('Position (m)')
ax.set_ylabel('Temps (s)')
ax.set_zlabel('Température (°C)')

plt.show()
