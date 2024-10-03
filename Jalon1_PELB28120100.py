import numpy as np
import matplotlib.pyplot as plt

# Définition des variables

<<<<<<< HEAD
k: float = 0.8  # Conductivité thermique du béton (W/m·K)
T_left: int = 100  # Température à la frontière gauche (°C) - Dirichlet
T_right_Dirichlet: int = 25  # Température à la frontière droite (°C) - Dirichlet
N: int = 1000  # Nombre de volumes finis
dx: float = L / N  # Taille des volumes finis
S: int = h * l 
=======
# Dimension de la barre en fer en mètre
L=10 # Longueur
e=1 # Largeur
h=1 # Hauteur
S=h*e
>>>>>>> 8f70f38ec3f153c1331946d490f3cc6fd6b06f29

# Conductivité thermique du fer
k=80 #à 20°C mais considéré comme constante ici

# Définition des coupe transversal
n=100
dx=0.1

# Conditions limites en degrés cas 1
Tg=100
Td=0

<<<<<<< HEAD
    for i in range(1, N-1):
       A[i, i - 1] = -(k * S) / dx
       A[i, i + 1] = -(k * S) / dx
       A[i, i] = 2 * (k * S) / dx 
    # Condition de Dirichlet à gauche
    A[0, 0] = (k * S) / (dx / 2) + (k * S) / dx  # calcul de la condition de Dirichlet à gauche
    A[0, 1] = -(k * S) / dx
    B[0] = (k * S) / (dx / 2) * T_left
    return A, B
=======
# Définition des variables de l'équation A*T=b
b=np.zeros(n)
A=np.zeros((n,n))

# Calcul mathématique
>>>>>>> 8f70f38ec3f153c1331946d490f3cc6fd6b06f29

for i in range(1,n-1):
    A[i, i - 1] = -(k * S) / dx
    A[i,i+1]=-(k*S)/dx
    A[i,i]=2*(k*S)/dx

<<<<<<< HEAD
    A[N - 1, N - 2] = -(k * S) / dx  # calcul de la condition de Dirichlet à droite
    A[N - 1, N - 1] = (k * S) / dx + (k * S) / (dx / 2)
    b[N - 1] = (k * S) / (dx / 2) * T_right_Dirichlet
=======
# Condition aux limites CAS 1 : Dirichlet 100°C à 0°C
A[0,0]=(k*S)/(dx/2)+(k*S)/dx
A[0,1]=-(k*S)/dx
b[0]=(k*S)/(dx/2) * Tg
>>>>>>> 8f70f38ec3f153c1331946d490f3cc6fd6b06f29

A[n-1,n-2]=-(k*S)/dx
A[n-1,n-1]=(k*S)/dx+(k*S)/(dx/2)
b[n-1]=(k*S)/(dx/2) * Td

T=np.linalg.solve(A,b)

#print(A)
#print (b)
print (T)

<<<<<<< HEAD
    # Condition de Neumann à droite
    # Résoudre le système linéaire Ax = 
    A[N - 1, N - 2] = -(k * S) / dx  # calcul de la condition de Neumann à droite
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
=======
#figure CAS 1
x1 = np.linspace(0, L, n)
plt.figure(figsize=(10, 8))
plt.plot(x1, T, label="Cas 1 : Conditions de Dirichlet", linestyle='-',color='blue')
plt.title("Graphe de la température en fonction de la distance")
plt.xlabel("longueur (m)")
plt.ylabel("Température (°C)")
>>>>>>> 8f70f38ec3f153c1331946d490f3cc6fd6b06f29
plt.grid(True)
plt.legend()
plt.show()
#+ on n est grand, + la précision est grande car la distance entre 2 centroïds est petites

# Condition aux limites CAS 2 : Dirichlet 100°C et condition de Neumann libre à droite
A[n-1,n-2]=-(k*S)/dx
A[n-1,n-1]=(k*S)/dx
b[n-1]=0

T=np.linalg.solve(A,b)

print(T)

#figure CAS 2
x1 = np.linspace(0, L, n)
plt.figure(figsize=(10, 8))
plt.plot(x1, T, label="Cas 1 : Conditions Dirichlet + Neumann", linestyle='-',color='blue')
plt.title("Graphe de la température en fonction de la distance")
plt.xlabel("longueur (m)")
plt.ylabel("Température (°C)")
plt.ylim([0, 120])
plt.grid(True)
plt.legend()
plt.show()
# hypothése negliger les erreurs 
