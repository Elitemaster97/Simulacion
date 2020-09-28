'''                             TAREA2
Diseña y ejecuta un experimento para determinar o (a) el mayor colapso
poblacional entre iteraciones subsecuentes o (b) el mayor tiempo
continuo de vida en una celda en una malla de 20 por 20 celdas hasta
que se mueran todas o que se hayan cumplido 50 iteraciones,
variando la probabilidad inicial de celda viva entre 0.1 y 0.9 en
pasos de 0.1 (en el ejemplo se distribuyen uniformemente al azar,
por lo cual la probabilidad es 0.5).
'''



import numpy as np # hay que instalar numpy a parte con pip3 o algo similar
from random import random
import matplotlib.cm as cm
import matplotlib.pyplot as plt


limite = range(50)          # iteracion limite
x = random()
y = .7



def give(x,y):
    x = random()
    if x < y:
        give = 1
    else:
        give = 0
    return give

rep = [d for d in np.arange(0.1, 1, 0.1)]
dim = 0
num = 0
valores = 0
actual = 0
vivos = 0
vivos2 = 0                      #Memoria de vivos
Porvivos = 0
Difvivos = 0
Difvivos2 = 0
graf = []               # vaiable para guradar datos a grficar




def mapeo(pos):
    fila = pos // dim
    columna = pos % dim
    return actual[fila, columna]

assert all([mapeo(x) == valores[x]  for x in range(num)])

def paso(pos):
    fila = pos // dim
    columna = pos % dim
    vecindad = actual[max(0, fila - 1):min(dim, fila + 2),
                      max(0, columna - 1):min(dim, columna + 2)]
    return 1 * (np.sum(vecindad) - actual[fila, columna] == 3)

for i in rep:
    y = i
    print("")
    print("-------------------------", i ,"-------------------------")
    dim = 20                #Tamaño de la matriz cuadrada
    num = dim**2
    valores = [give(x,y) for i in range(num)]
    actual = np.reshape(valores, (dim, dim)) # Rescalar valores en matriz
    vivos = sum(valores)
    Porvivos = (100 * (vivos / num))
    vivos2 = 0              #Condicion inical
    Difvivos = 0            #Condicion inical
    Difvivos2 = 0           #Condicion inical
    Maxvivos = 0            #Condicion inical
    print("Vivos:", vivos, "de", num, ", Porcentaje de sobrevivencia:", Porvivos, "%")
    for iteracion in limite:
        valores = [paso(x) for x in range(num)]
        Difvivos = vivos2 - vivos
        if Difvivos >= Maxvivos:
            Maxvivos = Difvivos
        print("Iter",iteracion, ", vivos:", vivos, ", vivos2", vivos2,", Difvivos", Difvivos, ", Maxvivos", Maxvivos)
        vivos2 = vivos
        vivos = sum(valores)
        if iteracion == max(limite):
            graf.append(Maxvivos)
        elif vivos == 0:
            graf.append(Maxvivos)
            print("        *** Para Iter",iteracion + 1, ' Ya no queda nadie vivo. ***')
            break
        actual = np.reshape(valores, (dim, dim))
    print("-------------------------", i ,"-------------------------")
    print("")
print(graf)
print("")

x = np.arange(0.1, 1, 0.1)
y = graf
plt.plot(x, y)
plt.xlabel('Probabilidad inicial de celda viva')
plt.ylabel('Mayor colapso poblacional')
plt.show()
plt.close()
