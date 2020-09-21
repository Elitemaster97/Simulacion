"""                           TAREA1
Examina de manera sistematica los efectos de la dimension en el tiempo
de regreso al origen del movimiento Browniano para dimensiones 1 a 8
en incrementos lineales de uno, variando el numero de pasos de la
caminata como potencias de dos con exponentes de 5 a 10 en incrementos
lienales de uno, con 50 repeticiones del experimento para cada
combinacion. Grafica los resultados en unsa sola figura con diagrama
de caja de bigote o violin, colocando y coloreando los resultados de
una distancia y otra de tal manera que es facil concluir si la
distancia utilizada tiene algun efecto en el comportamiento, o en su
defecto, incluir un cuadro indicando el minimo, proemdio maximo del
tiempo de regreso por cada dimension junto con el porcentaje de
caminatas que nunca regresaron.
"""

from random import random, randint
import matplotlib.pyplot as plt
import math

x = 4
y = 4
t = 0
INF = 0
cont = 0
tiempo = 0
PORINF = 0
DatosINF = []
DatosFIL = []

def contadory(y):
    if y < 10:
        y = y + 1
    else:
        y = 5
    return y

def contadorx(x,y):
    x = y
    if x < 10:
        x = x + 1
    else:
        x = 5
    x = 2**x
    return x

def paso(pos, dim):
    d = randint(0, dim - 1)     # Elegimos dimension
    pos[d] += -x if random() < 0.5 else x # Elegimos direccion y nos movemos
    return pos         # Actualizamos el paso realizado

def experimento(dim,tiempo, t, x, y):
    pos = [0] * dim     #Generando vector de posicion en el origen
    for o in range(pasos):     # Cantidad de pasos a realizar
        if t == 0:
            x = contadorx(x,y)       # Contador de 5 a 10
            y = contadory(y)        # Memoria del contador x
            pos = paso(pos, dim)     # Actualizamos pos
            if all([p == 0 for p in pos]):
                t = t + 1
            else:
                tiempo = tiempo + 1
                if tiempo == pasos:
                    tiempo = 0         # math.inf
    return tiempo + 1

def porinf(INF, total):
    l = -(100 * (INF / total)) + 100
    return l

rep = [d for d in range(1,20)]     # Dimensiones a probar   range(1,9)
total = 50           # Cantidad de experimentos a realizar
pasos = 70         # Cantidad de pasos a realiar
Datos = []


for i in rep:
    regresos = []       # Reset de variable
    dim = i
    for replica in range(total):
        regresos.append(experimento(dim, tiempo, t, x, y))
    #if dim == 1:
        #D1 = regresos
    #elif dim == 2:
        #D2 = regresos
    #elif dim == 3:
        #D3 = regresos
    #elif dim == 4:
        #D4 = regresos
    #elif dim == 5:
        #D5 = regresos
    #elif dim == 6:
        #D6 = regresos
    #elif dim == 7:
        #D7 = regresos
    #elif dim == 8:
        #D8 = regresos
    Datos.append(regresos)

#-------------------------FILTRO-------------------------
print("")
print("% Regreso al origen")
for dimencion in range(max(rep)):
    for filtro in range(total):
        cont = cont + 1
        if Datos[dimencion][filtro] == 1:
            INF = INF + 1
            if cont == total:
                DatosINF.append(DatosFIL)
                DatosFIL = []
                cont = 0
                PORINF = porinf(INF, total)
                print("Dimención", dimencion + 1,":" ,PORINF, "%")
                PORINF = 0
                INF = 0

        elif Datos[dimencion][filtro] != 1:
            DatosFIL.append(Datos[dimencion][filtro])
            if cont == total:
                DatosINF.append(DatosFIL)
                DatosFIL = []
                cont = 0
                PORINF = porinf(INF, total)
                print("Dimención", dimencion + 1,":" ,PORINF, "%")
                PORINF = 0
                INF = 0
print("")
#-------------------------GRAFICA-------------------------

plt.boxplot(DatosINF)
#plt.boxplot([D1, D2, D3, D4, D5, D6, D7, D8])
#plt.title('TAREA 1')
plt.xlabel('Dimensión')
plt.ylabel('Tiempo de regreso al origen')
plt.show()
plt.close()
