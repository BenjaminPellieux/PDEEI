import numpy as np
import matplotlib.pyplot as plt

# Définition des variables

# Dimension de la barre en fer en mètre
L=10 # Longueur
e=1 # Largeur
h=1 # Hauteur
S=h*e

# Conductivité thermique du fer
k=80 #à 20°C mais considéré comme constante ici

# Définition des coupe transversal
n=100
dx=0.1

# Conditions limites en degrés cas 1
Tg=100
Td=0

# Définition des variables de l'équation A*T=b
b=np.zeros(n)
A=np.zeros((n,n))

# Calcul mathématique

for i in range(1,n-1):
    A[i, i - 1] = -(k * S) / dx
    A[i,i+1]=-(k*S)/dx
    A[i,i]=2*(k*S)/dx

# Condition aux limites CAS 1 : Dirichlet 100°C à 0°C
A[0,0]=(k*S)/(dx/2)+(k*S)/dx
A[0,1]=-(k*S)/dx
b[0]=(k*S)/(dx/2) * Tg

A[n-1,n-2]=-(k*S)/dx
A[n-1,n-1]=(k*S)/dx+(k*S)/(dx/2)
b[n-1]=(k*S)/(dx/2) * Td

T=np.linalg.solve(A,b)

#print(A)
#print (b)
print (T)

#figure CAS 1
x1 = np.linspace(0, L, n)
plt.figure(figsize=(10, 8))
plt.plot(x1, T, label="Cas 1 : Conditions de Dirichlet", linestyle='-',color='blue')
plt.title("Graphe de la température en fonction de la distance")
plt.xlabel("longueur (m)")
plt.ylabel("Température (°C)")
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
