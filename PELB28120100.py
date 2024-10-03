import numpy as np
import matplotlib.pyplot as plt

# Paramètres physiques
L = 0.1  # Largeur --Longueur-- du mur (m)
k = 0.5  # Conductivité thermique du béton (W/m·K)
rho = 2400  # Densité du béton (kg/m^3)
c = 1050  # Capacité thermique spécifique du béton (J/g·K)
T_left = 30  # Température à la frontière gauche (°C) - Condition de Dirichlet
T_right = 15  # Température à la frontière droite (°C) - Condition de Dirichlet
T_consigne = 25  # Température de consigne pour le chauffage (°C)
S = 5000  # Source de chaleur (W/m^3)
T_init = 20

# Paramètres de discrétisation
N = 100  # Nombre de volumes finis
dx = L / N  # Taille des volumes finis
dt = 0.01  # Pas de temps (s)
alpha = k / (rho * c)  # Diffusivité thermique
t_max = 1000  # Temps de simulation (s)
n_steps = int(t_max / dt)  # Nombre de pas de temps

# Initialisation des tableaux de température
T = np.ones(N) * T_init  # Température initiale du mur
T_new = np.copy(T)



# Fonction de contrôle ON/OFF
def controle_on_off(T, S, T_consigne):
    if T < T_consigne:

        print(f'[LOG][INFO] Chaufage Activé \n{T=}\n{S=}\n{T_consigne=}') 
        return S  # Activer le chauffage
    else:
        print(f'[LOG][INFO] Chaufage Désactiver \n{T=}\n{S=}\n{T_consigne=}') 
        return 0  # Désactiver le chauffage

# Simulation
for t in range(n_steps):
    for i in range(1, N-1):
        # Terme source de chauffage interne avec contrôle ON/OFF
        source = controle_on_off(T[i], S, T_consigne)
        # Schéma explicite pour l'équation de la chaleur
        T_new[i] = T[i] + alpha * dt / dx**2 * (T[i+1] - 2*T[i] + T[i-1]) + dt * source / (rho * c)
    
    # Conditions aux limites de Dirichlet
    T_new[0] = T_left
    T_new[-1] = T_right
    
    # Mise à jour des températures
    T = np.copy(T_new)

# Affichage des résultats
x = np.linspace(0, L, N)
plt.plot(x, T, marker='o')
plt.title(f'Temperature distribution after {t_max} seconds')
plt.xlabel('Position (m)')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.show()

