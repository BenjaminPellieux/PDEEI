import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import diags
from scipy.linalg import solve_banded
from datetime import datetime


# TODO: ADD GEOMETRIE MULTICOUCHE

##########################################
#                Constantes              #
##########################################

T_base = 10
YEAR = 2023
MONTH = 7
DAY = 6
file_path = f"data/WD__{YEAR}.csv"
Form_liste = ["Dirichlet-Dirichlet","Dirichlet-Neuman"]
TODAY = datetime.today().strftime('%m-%d')


###################################
#        Donnée du mur            #
###################################

L = 0.5  # longeur Dimension = du mur (m)
l = 0.2  # Largeur 
h = 0.2 # Hauteur
S = h * l  # Surface
N = 1000  # Nombre de volumes finis
dx = L / N  # Taille des volumes finis
v = dx * S  # Volume

# 1/6 beton // 1/6 air // 4/6 beton

###################################
#       Donnée des temperatures   #
###################################

temperature_data = pd.read_csv(file_path)
temperature_data['Date'] = pd.to_datetime(temperature_data['Date'], format='%Y/%m/%d %I:%M %p')
temperature_data.set_index('Date', inplace=True)

# Extract temperature data
temperature_data['Temperature (°C)'] = temperature_data['Temperature (°C)'].astype(float)

T_left = None  # Température à la frontière gauche (°C) - Dirichlet
T_right = 21  # Température à la frontière droite (°C) - Dirichlet
src = 0  # Source de chaleur au centre

###################################
#       Donnée du Béton           #
###################################

k_beton = 1.5  # Conductivité thermique du béton (W/m·K)
p_beton = 2200  # kg/m3
c_beton = 880  # (J/K·kg)

###################################
#       Donnée de l'air           #
###################################

k_air = 0.025  # Conductivité thermique de l'air (W/m·K)
p_air = 1.204   # kg/m3
c_air = 1004  # (J/K·kg)

###################################
#     Parametre de Simulation     #
###################################

# 1/6 beton // 1/6 air // 4/6 beton
comp_mur = [round(N / 6), round(N / 6), round(N * 4 / 6 ) - 1]
k_values = np.array([k_beton] * comp_mur[0] + [k_air] * comp_mur[1] + [k_beton] * comp_mur[2])
c_values = np.array([c_beton] * comp_mur[0] + [c_air] * comp_mur[1] + [c_beton] * comp_mur[2])
p_values = np.array([p_beton] * comp_mur[0] + [p_air] * comp_mur[1] + [p_beton] * comp_mur[2])

heures = 72
t_total = 3600 * heures  # Simulation sur X heures
dt = 900  # Intervalle de temps en secondes (15 minutes)

T_old = np.ones(N) * 20  # Température initiale
T_new = np.copy(T_old)

all_temperatures = []

##########################################
#         Fonctions de simulation        #
##########################################
def init_tridiagonal_matrix() -> (np.ndarray, np.ndarray):
    """Initialisation de la matrice tridiagonale et du vecteur B avec plusieurs couches."""
    # Préparation des coefficients pour chaque couche
    # Initialisation des diagonales
    lower_diag = np.zeros(N - 1)
    main_diag = np.zeros(N)
    upper_diag = np.zeros(N - 1)
    B = np.copy(T_old)

   
    # Remplissage des coefficients pour chaque volume fini
    for i in range(1, N - 1):
        k_eff = (k_values[i - 1] + k_values[i]) / 2 # Conductivité effective entre deux volumes adjacents
        lower_diag[i - 1] = -k_eff * S / dx
        upper_diag[i] = -k_eff * S / dx
        main_diag[i] = 2 * k_eff * S / dx + (p_values[i] * c_values[i] * v) / dt

    # Conditions aux limites Dirichlet à gauche
    main_diag[0] = (k_values[0] * S) / (dx / 2) + (k_values[0] * S) / dx
    B[0] = (k_values[0] * S) / (dx / 2) * T_left + T_old[0]

    # Source de chaleur au centre
    B[N // 2] += src

    return lower_diag, main_diag, upper_diag, B


def solve_dirichlet_dirichlet() -> np.ndarray:
    """ Résolution du problème avec conditions de Dirichlet-Dirichlet """
    lower_diag, main_diag, upper_diag, B = init_tridiagonal_matrix()

    # Condition Dirichlet à droite
    main_diag[-1] = (k_values[-1] * S) / dx + (k_values[-1] * S) / (dx / 2) + ((p_values[-1] * c_values[-1] * v) / dt)
    B[-1] = (k_values[-1] * S) / (dx / 2) * T_right + T_old[-1]

    ab = np.zeros((3, N))
    ab[0, 1:] = upper_diag  # Surdiagonale
    ab[1, :] = main_diag    # Diagonale principale
    ab[2, :-1] = lower_diag  # Sous-diagonale

    return solve_banded((1, 1), ab, B)

def solve_dirichlet_neumann() -> np.ndarray:
    """ Résolution du problème avec conditions Dirichlet-Neumann """
    lower_diag, main_diag, upper_diag, B = init_tridiagonal_matrix()

    # Condition Neumann à droite
    main_diag[-1] = (k_values[-1] * S) / dx + ((p_values[-1] * c_values[-1] * v) / dt)
    B[-1] = T_old[-1]

    ab = np.zeros((3, N))
    ab[0, 1:] = upper_diag  # Surdiagonale
    ab[1, :] = main_diag    # Diagonale principale
    ab[2, :-1] = lower_diag  # Sous-diagonale

    return solve_banded((1, 1), ab, B)

##########################################
#           Boucle de simulation         #
##########################################

x = np.linspace(0, L, N)
temps = []
choix = int(input("Bonjour bienvenu dans la simulation de Benjamin PELLIEUX.\nEntrez votre choix: \n1: Dirichlet-Dirichlet\n2: Dirichlet-Neumann\n "))
if choix == 1:
    print(f"[INFO] Running Dirichlet-Dirichlet")
else:
    print(f"[INFO] Running Dirichlet-Neumann simunation ")
print(f"[INFO] Durée en seconde {t_total}s")
print(f"[INFO] Nombre d'éléments finis : {N} \n {k_values.shape}")


for i in range(0, t_total, dt):

    current_time = datetime(YEAR, MONTH, DAY) + pd.Timedelta(hours= (i // 3600) % 24)
    try:
        T_left = temperature_data.loc[current_time, 'Temperature (°C)']
    except KeyError:
        print(f"[ERROR] No temperature data available for time: {current_time}")
        T_left = 0  # fallback value in case of missing data

    for j in range(N):
        T_old[j] = T_new[j] * ((c_values[j] * p_values[j] * v) / dt)

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

all_temperatures = np.array(all_temperatures)  # (temps, position)
temps = np.array(temps)
X, Y = np.meshgrid(x, temps)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Afficher la température en fonction du temps et de la position
ax.plot_surface(X, Y, all_temperatures, cmap='plasma')

ax.set_title(f'Évolution de la température dans le mur \n avec la condition {Form_liste[choix-1]}.\n Durée de simulation : {heures}h')
ax.set_xlabel('Position (m)')
ax.set_ylabel('Temps (s)')
ax.set_zlabel('Température (°C)')

plt.show()
