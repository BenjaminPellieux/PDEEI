import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib.animation import FuncAnimation

##########################################
#                Constantes              #
##########################################

# Animation
FPS = 60  # Fréquence d'images par seconde pour l'animation
GIF = True  # Si True, enregistre l'animation en GIF

###################################
#        Données géométriques     #
###################################

L = 0.30  # Longueur du mur (m)
l = 3  # Largeur du mur (m)
h = 2  # Hauteur du mur (m)
S = h * l  # Surface du mur (m²)

###################################
#       Données thermiques        #
###################################

# Températures initiales et limites
T_init = 12  # Température initiale dans le mur (°C)
T_left = 25  # Température à la frontière gauche (°C) - Dirichlet
T_cible = 18 # Température cible pour le mur de droite

# Propriétés des matériaux
k_beton, c_beton, p_beton = 1.5, 880, 2200  # Béton standard
k_air, c_air, p_air, h_air= 0.026, 1004, 1.204, 20  # Air
power_beton_conducteur = 3000  # Puissance thermique dans la couche chauffante (W)

###################################
#     Parametre de Simulation     #
###################################

# Distribution des couches dans le mur:  1/6 beton // 1/6 air // 1/20 beton conducteur et le reste de beton +- 4/6
NVF_beton1, NVF_air, NVF_beton_conducteur, NVF_beton2 = 200, 500, 10, 1000  # Nombre de volumes finis

NVF_tot = NVF_beton1 + NVF_air + NVF_beton_conducteur + NVF_beton2 
comp_mur = [round(NVF_tot / 6), round(NVF_tot / 6), round(NVF_tot / 20)]
NVF_beton2 = NVF_tot - sum(comp_mur)
comp_mur.append(NVF_beton2)
 
# Propriétés thermiques par couche
k_values = np.array([k_beton] * comp_mur[0] + 
                    [k_air]   * comp_mur[1] + 
                    [k_beton] * comp_mur[2] + 
                    [k_beton] * comp_mur[3])

c_values = np.array([c_beton] * comp_mur[0] + 
                    [c_air]   * comp_mur[1] + 
                    [c_beton] * comp_mur[2] + 
                    [c_beton] * comp_mur[3])

p_values = np.array([p_beton] * comp_mur[0] + 
                    [p_air]   * comp_mur[1] + 
                    [p_beton] * comp_mur[2] + 
                    [p_beton] * comp_mur[3])

# Dimensions des volumes finis
dx_values = np.array([L / (6 * NVF_beton1)] * comp_mur[0] + 
                     [L / (6 * NVF_air)] * comp_mur[1] + 
                     [L / (20 * NVF_beton_conducteur)] * comp_mur[2] + 
                     [L / NVF_beton2] * comp_mur[3]
                     ) 
v_values = dx_values * S

# Paramètres temporels
heures = 48  # Durée de la simulation (heures)
t_total = 3600 * heures  # Simulation sur X heures
dt = 1800  # Intervalle de temps en secondes (30 minutes)

T_old = np.ones(NVF_tot) * T_init  # Température initiale
T_new = np.copy(T_old)
src = np.zeros(NVF_tot)
src[NVF_beton1 + NVF_air : NVF_beton1 + NVF_air + NVF_beton_conducteur] = power_beton_conducteur / NVF_beton_conducteur

# Stocker toutes les températures pour chaque étape de temps
all_temperatures = []

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

        k_eff_lower = (((dx_values[i] * k_values[i] + dx_values[i - 1] * k_values[i - 1]) * S) / (dx_values[i] + dx_values[i - 1])) /  ((dx_values[i] + dx_values[i - 1]) / 2)
        k_eff_upper = (((dx_values[i] * k_values[i] + dx_values[i + 1] * k_values[i + 1]) * S) / (dx_values[i] + dx_values[i + 1])) /  ((dx_values[i] + dx_values[i + 1]) / 2)
        
        A[i, i - 1], A[i, i + 1] = -k_eff_lower, -k_eff_upper
        A[i, i] = k_eff_lower + k_eff_upper + (p_values[i] * c_values[i] * v_values[i]) / dt
        
        if float(T_old[-1]) < T_cible:
            B[i] += src[i] * ((c_values[i] * p_values[i] * v_values[i]) / dt)

    # Conditions aux limites
    k_eff_upper = (((dx_values[0] * k_values[0] + dx_values[1] * k_values[1]) * S) / (dx_values[0] + dx_values[1])) /  ((dx_values[0] + dx_values[1]) / 2)
    A[0, 1] = -k_eff_upper
    A[0, 0], B[0] = h_air * S +  k_eff_upper + ((c_values[0] * p_values[0] * v_values[0]) / dt), h_air * S  * T_left + T_old[0]  

    k_eff_lower = (((dx_values[-1] * k_values[-1] + dx_values[-2] * k_values[-2]) * S) / (dx_values[-1] + dx_values[-2])) /  ((dx_values[-1] + dx_values[-2]) / 2)
    A[-1, -2] = -k_eff_lower
    A[-1, -1], B[-1] = k_eff_lower + ((p_values[-1] * c_values[-1] * v_values[-1]) / dt), T_old[-1]
    
    return np.linalg.solve(A, B)


    
def init_ani():
    line.set_data([], [])
    time_text.set_text('') 
    return line, time_text

def update_ani(frame):
    T_new = all_temperatures[frame]
    line.set_data(x, T_new)
    time_text.set_text(f'Heure de simulation : {frame * dt / 3600:.2f}h \n Température intérieur {float(T_new[-1]):.2f}°C')
    return line, time_text


##########################################
#           Boucle de simulation         #
##########################################
x = [(dx_values[0] / 2)]
for i in range(NVF_tot - 1):
    x.append(x[-1] + ((dx_values[i] + dx_values[i + 1]) / 2))


print("Bonjour bienvenu dans la simulation de Benjamin PELLIEUX Mixte - Neumann")
print(f"[INFO] Running Mixte - Neumann")

print(f"[INFO] Durée en seconde {t_total}s")
print(f"[INFO] Nombre d'éléments finis : {NVF_tot}")

for _ in range(0, t_total, dt):

    for j in range(NVF_tot):
        T_old[j] = T_new[j] * ((c_values[j] * p_values[j] * v_values[j]) / dt)
    T_new = solve_mixte_neumann()
    all_temperatures.append(T_new.copy())

##########################################
#         Affichage des résultats        #
##########################################

fig, ax = plt.subplots(figsize=(10,10))
line, = ax.plot([], [], "-", color="red", lw=2, label="Température")
ax.set_xlim(0, L)
ax.set_ylim(0, 30)
ax.set_xlabel("Position (m)")
ax.set_ylabel("Température (°C)")
ax.set_title(f"Évolution de la température dans le mur en fonction du temps\n Temparture cible: {T_cible}°C")
ax.legend(loc="upper left")

time_text = ax.text(0.02, 0.85, '', transform=ax.transAxes, color="black", fontsize=12, ha="left")

ani = FuncAnimation(fig, update_ani, frames=len(all_temperatures), init_func=init_ani, blit=True, repeat=True)

if GIF:
    ani.save("Simulation_Jalon3.gif", writer='pillow', fps=FPS)
    print("L'animation a été enregistrée sous le nom Simulation_Jalon3.gif")
    
plt.show()