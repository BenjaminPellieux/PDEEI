"""
Date: 18/11/2024
Author: Benjamin PELLIEUX <bpellieux@etu.uqac.ca>
Status 
Simulation de la conduction thermique dans un mur multicouche avec une couche chauffante

Ce script modélise la diffusion thermique dans un mur en béton comprenant plusieurs couches,
avec des propriétés thermiques distinctes pour chaque matériau (béton, air, béton conducteur).
Une source de chaleur est ajoutée dans une couche de béton conducteur pour simuler un chauffage.

Le script produit une animation montrant l'évolution de la température dans le mur en fonction
du temps et de la position.

Bibliothèques utilisées :
- numpy : Calculs numériques et gestion des tableaux
- matplotlib : Visualisation des données et animation
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

##########################################
#                Constantes              #
##########################################

# Animation
FPS: int = 60  # Fréquence d'images par seconde pour l'animation
GIF: bool = False  # Si True, enregistre l'animation en GIF

###################################
#        Données géométriques     #
###################################

L: float = 0.30  # Longueur du mur (m)
l: int = 3  # Largeur du mur (m)
h: int = 2  # Hauteur du mur (m)
S: int = h * l  # Surface du mur (m²)

###################################
#       Données thermiques        #
###################################

# Températures initiales et limites
T_init: int = 10  # Température initiale dans le mur (°C)
T_left: int = 23  # Température à la frontière gauche (°C) - Dirichlet
T_cible: int = 18 # Température cible pour le mur de droite

# Propriétés des matériaux
k_beton, c_beton, p_beton = 1.28, 880, 2200  # Béton standard
k_air, c_air, p_air, h_air = 0.026, 1004, 1.204, 20  # Air
power_beton_conducteur: int = 200  # Puissance thermique dans la couche chauffante (W)

###################################
#     Parametre de Simulation     #
###################################

# Distribution des couches dans le mur:  1/6 beton // 1/6 air // 1/20 beton conducteur et le reste de beton +- 4/6
NVF_beton1, NVF_air, NVF_beton_conducteur, NVF_beton2 = 800, 800, 100, 600  # Nombre de volumes finis


print(f"{NVF_beton2=}")
NVF_tot: int = NVF_beton1 + NVF_air + NVF_beton_conducteur + NVF_beton2 
comp_mur: list[float] = [round(NVF_tot / 6), round(NVF_tot / 6), round(NVF_tot / 20), round((NVF_tot * 37) /60)]
comp_mur[-1] += abs(NVF_tot - sum(comp_mur))
# NVF_beton2 = NVF_tot - sum(comp_mur)
# comp_mur.append(NVF_beton2)
 

print(f"{comp_mur=}")
# Propriétés thermiques par couche
k_values: np.ndarray[float] = np.array(
                    [k_beton] * comp_mur[0] + 
                    [k_air]   * comp_mur[1] + 
                    [k_beton] * comp_mur[2] + 
                    [k_beton] * comp_mur[3])

c_values: np.ndarray[float] = np.array(
                    [c_beton] * comp_mur[0] + 
                    [c_air]   * comp_mur[1] + 
                    [c_beton] * comp_mur[2] + 
                    [c_beton] * comp_mur[3])

p_values: np.ndarray[float] = np.array(
                    [p_beton] * comp_mur[0] + 
                    [p_air]   * comp_mur[1] + 
                    [p_beton] * comp_mur[2] + 
                    [p_beton] * comp_mur[3])

# Dimensions des volumes finis
dx_values: np.ndarray[float] = np.array(
                     [L / (6 * NVF_beton1)] * comp_mur[0] + 
                     [L / (6 * NVF_air)] * comp_mur[1] + 
                     [L / (20 * NVF_beton_conducteur)] * comp_mur[2] + 
                     [(37 * L) / (60 * NVF_beton2)] * comp_mur[3]
                     ) 
v_values = dx_values * S

# Paramètres temporels
HEURES: int = 48  # Durée de la simulation (heures)
t_total: int = 3600 * HEURES  # Simulation sur X heures
dt: int = 1800  # Intervalle de temps en secondes

T_old: np.ndarray[float] = np.ones(NVF_tot) * T_init  # Température initiale
T_new: np.ndarray[float] = np.copy(T_old)
src: np.ndarray[float] = np.zeros(NVF_tot)
src[NVF_beton1 + NVF_air : NVF_beton1 + NVF_air + NVF_beton_conducteur] = power_beton_conducteur / NVF_beton_conducteur

# Stocker toutes les températures pour chaque étape de temps
all_source_status: list[bool] = []
all_temperatures: list[np.ndarray] = []

##########################################
#         Fonctions de simulation        #
##########################################

def solve_mixte_neumann() -> np.ndarray:
    """
    Résolution du problème avec conditions aux limites mixtes (Dirichlet-Neumann).
    Retourne la nouvelle distribution de température dans le mur.
    """
    
    A, B = np.zeros((NVF_tot, NVF_tot)), np.copy(T_old)

    # Construction de la matrice et du vecteur B
    for i in range(1, NVF_tot - 1):

        k_eff_lower: float = (((dx_values[i] * k_values[i] + dx_values[i - 1] * k_values[i - 1]) * S) / (dx_values[i] + dx_values[i - 1])) /  ((dx_values[i] + dx_values[i - 1]) / 2)
        k_eff_upper: float = (((dx_values[i] * k_values[i] + dx_values[i + 1] * k_values[i + 1]) * S) / (dx_values[i] + dx_values[i + 1])) /  ((dx_values[i] + dx_values[i + 1]) / 2)
        
        A[i, i - 1], A[i, i + 1] = -k_eff_lower, -k_eff_upper
        A[i, i] = k_eff_lower + k_eff_upper + (p_values[i] * c_values[i] * v_values[i]) / dt
        
        if float(T_new[-1]) < T_cible:
            B[i] += src[i] * ((c_values[i] * p_values[i] * v_values[i]) / dt)

    # Conditions aux limites
    k_eff_upper: float = (((dx_values[0] * k_values[0] + dx_values[1] * k_values[1]) * S) / (dx_values[0] + dx_values[1])) /  ((dx_values[0] + dx_values[1]) / 2)
    A[0, 1] = -k_eff_upper
    A[0, 0], B[0] = h_air * S +  k_eff_upper + ((c_values[0] * p_values[0] * v_values[0]) / dt), h_air * S  * T_left + T_old[0]  

    k_eff_lower: float = (((dx_values[-1] * k_values[-1] + dx_values[-2] * k_values[-2]) * S) / (dx_values[-1] + dx_values[-2])) /  ((dx_values[-1] + dx_values[-2]) / 2)
    A[-1, -2] = -k_eff_lower
    A[-1, -1], B[-1] = k_eff_lower + ((p_values[-1] * c_values[-1] * v_values[-1]) / dt), T_old[-1]
    
    return np.linalg.solve(A, B)


    
def init_ani():
    line.set_data([], [])
    time_text.set_text('') 
    return line, time_text

def update_ani(frame):
    T_new = all_temperatures[frame]
    source_status = all_source_status[frame]
    line.set_data(x, T_new)
    time_text.set_text(f" Heure de simulation : {frame * dt / 3600:.0f}h \n Température intérieur {float(T_new[-1]):.2f}°C\n Statut de l'élement chauffant: {'Activé' if source_status else 'Désactivé'}")
    return line, time_text


##########################################
#           Boucle de simulation         #
##########################################
x: list[float] = [(dx_values[0] / 2)]
for i in range(NVF_tot - 1):
    x.append(x[-1] + ((dx_values[i] + dx_values[i + 1]) / 2))


print("Bonjour bienvenu dans la simulation de Benjamin PELLIEUX Mixte - Neumann")
print(f"[INFO] Running Mixte - Neumann")

print(f"[INFO] Durée de la simulation : {(t_total / 3600):.0f}h -> {t_total}s")
print(f"[INFO] Nombre d'éléments finis : {NVF_tot}")
print(f"[INFO] Intervalle de temps en secondes {dt}s  {int(dt/60)} minutes")


for _ in range(0, t_total, dt):

    for j in range(NVF_tot):
        T_old[j] = T_new[j] * ((c_values[j] * p_values[j] * v_values[j]) / dt)
    T_new = solve_mixte_neumann()
    
    all_source_status.append(float(T_new[-1]) < T_cible)
    all_temperatures.append(T_new.copy())

##########################################
#         Affichage des résultats        #
##########################################

fig, ax = plt.subplots(figsize=(10,10))
line, = ax.plot([], [], "-", color="red", lw=2, label="Température")
ax.set_xlim(0, L)
ax.set_ylim(-5, 35)
ax.set_xlabel("Position (m)")
ax.set_ylabel("Température (°C)")
ax.set_title(f"Évolution de la température dans le mur en fonction du temps \navec un élément chauffant de {power_beton_conducteur}W et une temparture cible: {T_cible}°C")
ax.legend(loc="upper left")

time_text = ax.text(0.00, 0.85, '', transform=ax.transAxes, color="black", fontsize=12, ha="left")

ani = FuncAnimation(fig, update_ani, frames=len(all_temperatures), init_func=init_ani, blit=True, repeat=True)

if GIF:
    ani.save(f"Simulation_Jalon3_{HEURES}H.gif", writer='pillow', fps=FPS)
    print(f"L'animation a été enregistrée sous le nom Simulation_Jalon3_{HEURES}H.gif")
    
plt.show()