import numpy as np
import matplotlib.pyplot as plt
# https://fr.wikipedia.org/wiki/Liste_de_conductivit%C3%A9s_thermiques
# Paramètres physiques
L: int = 1  # Dimension = du mur (m)
l: int = 1
h: int = 1
k: float = 80  #0.8  # Conductivité thermique du béton (W/m·K)
T_left: int = 100  # Température à la frontière gauche (°C) - Dirichlet
T_right_Dirichlet: int = 25  # Température à la frontière droite (°C) - Dirichlet
N: int = 1000  # Nombre de volumes finis
dx: float = L / N  # Taille des volumes finis
S: int = h * l # S = A
src = 100
v = dx * S
p = 7800  #2200  #kg / m3
c = 450 #880 #(J K−1 kg−1) 
T = 0
t_total = 3600 * 2
nt = 1000
dt = 1
T_1 = 20
T_old = [0 for _ in range(N)]
T_new = [20 for _ in range(N)]



def init_matrix() -> [np.ndarray, np.ndarray]: 
    """ Construction de la matrice du système """
    A, B = np.zeros((N, N)), np.zeros(N)
    for i in range(1, N-1):
        A[i, i - 1] = -(k * S) / dx
        A[i, i + 1] = -(k * S) / dx
        A[i, i] = 2 * (k * S) / dx + ((p * c * v) /dt) #AE + AW + AT 
        B[i] = T_old[i]
    
    # Condition de Dirichlet à gauche
    A[0, 0] = (k * S) / (dx / 2) + (k * S) / dx 
    A[0, 1] = -(k * S) / dx
    B[0] = (k * S) / (dx / 2) * T_left
    B[N//2] += src
    B[0] = (k * S) / (dx / 2) * T_left + T_old[0]  

    return A, B


def solve_dirichlet(i: int = 0) -> np.ndarray:
    
    """ Fonction pour résoudre le problème avec conditions 
    de Dirichlet aux deux extrémités"""
    A, b = init_matrix()

    A[N - 1, N - 2] = -(k * S) / dx 
    A[N - 1, N - 1] = (k * S) / dx + (k * S) / (dx / 2) + ((p * c * v) / dt)
    b[N - 1] = (k * S) / (dx / 2) * T_right_Dirichlet + T_old[-1]
    
    
    T: np.ndarray = np.linalg.solve(A, b)
    T_new = T
    return T

def solve_dirichlet_neumann(i: int = None) -> np.ndarray:
    """ Fonction pour résoudre le problème avec condition 
    de Dirichlet à gauche et Neumann à droite """
    A, b  = init_matrix()
    
    A[N - 1, N - 2] = -(k * S) / dx  
    A[N - 1, N - 1] = (k * S) / dx + ((p * c * v) / dt)
    b[N - 1] =  T_old[-1]  
    
    T: np.ndarray = np.linalg.solve(A, b)
    T_new = T
    return T


# Résolution des deux scénarios
# T_dirichlet: np.ndarray = solve_dirichlet()
# T_dirichlet_neumann: np.ndarray = solve_dirichlet_neumann()

x = np.linspace(0, L, N)
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Horizontally stacked subplots')
plt.ylim(0, 200)

choix =  int(input("Entrez choix: \n1: Dirichlet-Dirichlet\n2: Dirichlet-Neuman\n "))
for i in range(t_total):
    
    #print(f"[DEBUG] {i}")
    for i in range(N):
        T_old[i] = T_new[i] * ((c * p * v) / dt)

    if choix == 1:
        
        ax1.plot(x, solve_dirichlet(i), label='Dirichlet')
    else:
        ax2.plot(x, solve_dirichlet_neumann(i),  label='Dirichlet-Neumann')

# Affichage des résultats


plt.title('Distribution de la température dans le mur ')
plt.xlabel('Position (m)')
plt.ylabel('Température (°C)')
#plt.legend()
plt.grid(True)
plt.show()
