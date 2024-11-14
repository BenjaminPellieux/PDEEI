import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import diags
from scipy.linalg import solve_banded
from datetime import datetime
from matplotlib.animation import FuncAnimation

# TODO: ADD GEOMETRIE MULTICOUCHE
# TODO: ADD ELEMENT CHAUFFANT 

##########################################
#                Constantes              #
##########################################

FPS = 30
GIF = True
Temp_auto = False
T_base = 10
YEAR = 2023
MONTH = 7
DAY = 6
file_path = f"data/WD__{YEAR}.csv"
TODAY = datetime.today().strftime('%m-%d')


###################################
#        Donnée du mur            #
###################################

L = 0.30  # longeur Dimension = du mur (m)
l = 3  # Largeur 
h = 2 # Hauteur
S = h * l  # Surface

# 1/6 beton // 1/6 air // 4/6 beton

###################################
#       Donnée des temperatures   #
###################################

temperature_data = pd.read_csv(file_path)
temperature_data['Date'] = pd.to_datetime(temperature_data['Date'], format='%Y/%m/%d %I:%M %p')
temperature_data.set_index('Date', inplace=True)

# Extract temperature data
temperature_data['Temperature (°C)'] = temperature_data['Temperature (°C)'].astype(float)

T_init = 10
T_left = 35  # Température à la frontière gauche (°C) - Dirichlet
src = 0  # Source de chaleur au centre

###################################
#       Donnée du Béton           #
###################################


NVF_beton1 = 200
NVF_beton2 = 800 #Nombre de Volumes Finis
k_beton = 1.5  # Conductivité thermique du béton (W/m·K)
p_beton = 2200  # kg/m3
c_beton = 880  # Capacitée thermique du beton (J/K·kg)

###################################
#       Donnée de l'air           #
###################################

NVF_air = 100
h_air = 20 # Convection normal de l'aire  
k_air = 2  #0.025  # Conductivité thermique de l'air (W/m·K)
p_air = 1.204   # kg/m3
c_air = 1004  # Capacitée thermique de l'air (J/K·kg)

###################################
#     Parametre de Simulation     #
###################################

# 1/20 beton conducteur  dans 3 eme section

# 1/6 beton // 1/6 air // 4/6 beton
NVF_tot = NVF_beton1 + NVF_beton2 + NVF_air
end = abs(NVF_tot - (round(NVF_tot / 6) + round(NVF_tot / 6) + round(NVF_tot * 4 / 6 ))) 

comp_mur = [round(NVF_tot / 6), round(NVF_tot / 6), round(NVF_tot * 4 / 6 ) + end]
k_values = np.array([k_beton] * comp_mur[0] + [k_air] * comp_mur[1] + [k_beton] * comp_mur[2])
c_values = np.array([c_beton] * comp_mur[0] + [c_air] * comp_mur[1] + [c_beton] * comp_mur[2])
p_values = np.array([p_beton] * comp_mur[0] + [p_air] * comp_mur[1] + [p_beton] * comp_mur[2])
dx_values = np.array([L / (6 * NVF_beton1)] * comp_mur[0] + [L / (6 * NVF_air)] * comp_mur[1] + [ (4 * L) / (6 * NVF_beton2)] * comp_mur[2]) 
v_values = dx_values * S


print(f"[DEBUG] {NVF_tot=} {end=}\n {len(k_values)=}\n {len(c_values)=}\n {len(p_values)=}\n {len(dx_values)=}")


heures = 24
t_total = 3600 * heures  # Simulation sur X heures
dt = 600  # Intervalle de temps en secondes (15 minutes)

T_old = np.ones(NVF_tot) * T_init  # Température initiale
T_new = np.copy(T_old)

# TODO: Neumann NOT DIRIRI
# TODO: Element chauffant beton conduteur 

##########################################
#         Fonctions de simulation        #
##########################################

def solve_dirichlet_neumann() -> np.ndarray:
    """ Résolution du problème avec conditions Dirichlet-Neumann """
    
    A, B = np.zeros((NVF_tot, NVF_tot)), np.copy(T_old)

    for i in range(1, NVF_tot - 1):

        # Conductivité harmonique entre deux volumes adjacents
        k_eff_lower = (((dx_values[i] * k_values[i] + dx_values[i - 1] * k_values[i - 1]) * S) / (dx_values[i] + dx_values[i - 1])) /  ((dx_values[i] + dx_values[i - 1]) / 2)
        k_eff_upper = (((dx_values[i] * k_values[i] + dx_values[i + 1] * k_values[i + 1]) * S) / (dx_values[i] + dx_values[i + 1])) /  ((dx_values[i] + dx_values[i + 1]) / 2)
        A[i, i - 1], A[i, i + 1] = -k_eff_lower, -k_eff_upper
        A[i, i] = k_eff_lower + k_eff_upper + (p_values[i] * c_values[i] * v_values[i]) / dt
    
    # Conditions aux limites mixte à gauche
    k_eff_upper = (((dx_values[0] * k_values[0] + dx_values[0 + 1] * k_values[0 + 1]) * S) / (dx_values[0] + dx_values[0 + 1])) /  ((dx_values[0] + dx_values[0 + 1]) / 2)
    A[0, 1] = -k_eff_upper
    A[0, 0], B[0] = h_air * S +  k_eff_upper + ((c_values[0] * p_values[0] * v_values[0]) / dt), h_air * S  * T_left + T_old[0]  

    # Condition Neumann à droite
    k_eff_lower = (((dx_values[-1] * k_values[-1] + dx_values[-2] * k_values[-2]) * S) / (dx_values[-1] + dx_values[-2])) /  ((dx_values[-1] + dx_values[-2]) / 2)
    A[-1, -2] = -k_eff_lower
    A[-1, -1], B[-1] = k_eff_lower + ((p_values[-1] * c_values[-1] * v_values[-1]) / dt), T_old[-1]
    
    
    # Source de chaleur au centre
    # B[N // 2] += src

    return np.linalg.solve(A, B)


def init_ani():
    line.set_data([], [])
    return line,

# Mettre à jour la température pour chaque image
def update_ani(frame):
    global T_new, T_old
    if Temp_auto:
        current_time = datetime(YEAR, MONTH, DAY) + pd.Timedelta(hours= (i // 3600) % 24)

        try:
            T_left = temperature_data.loc[current_time, 'Temperature (°C)']
        except KeyError:
            print(f"[ERROR] No temperature data available for time: {current_time}")
            T_left = 0  # fallback value in case of missing data


    for j in range(NVF_tot - end):
        T_old[j] = T_new[j] * ((c_values[j] * p_values[j] * v_values[j]) / dt)

    T_new = solve_dirichlet_neumann()

    # Mise à jour des données de la ligne pour chaque frame
    line.set_data(x, T_new)
    return line,

##########################################
#           Boucle de simulation         #
##########################################
x = [(dx_values[0] / 2)]
for i in range(NVF_tot - 1):
    x.append(x[-1] + ((dx_values[i] + dx_values[i + 1]) / 2))

# x = np.linspace(0, L, N)
temps = []
print("Bonjour bienvenu dans la simulation de Benjamin PELLIEUX Dirichlet-Neumann")
print(f"[INFO] Running Dirichlet-Dirichlet")

print(f"[INFO] Durée en seconde {t_total}s")
print(f"[INFO] Nombre d'éléments finis : {NVF_tot}")


##########################################
#         Affichage des résultats        #
##########################################


temps = np.array(temps)
X, Y = np.meshgrid(x, temps)


fig, ax = plt.subplots(figsize=(10,10))
line, = ax.plot([], [], "--", color="blue", lw=1, label=f"Courbe de température à l'intérieur du mur")
ax.set_xlim(0, L)
ax.set_ylim(0, 50)  # Ajuster l'échelle de température si besoin
ax.set_xlabel("Position (m)")
ax.set_ylabel("Température (°C)")
ax.set_title("Évolution de la température dans le mur en fonction du temps")
ax.legend(loc="upper left")


ani = FuncAnimation(fig, update_ani, frames=range(0, t_total, dt), init_func=init_ani, blit=True, repeat=False)

if GIF:
    ani.save("Simulation_Jalon3.gif", writer='pillow', fps=FPS)
    print(f"L'animation a été enregistrée sous le nom Simulation_Jalon3.gif")

plt.show()