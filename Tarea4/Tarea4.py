'''                             TAREA4
Examina de manera sistemática el efecto del número de semillas
y del tamaño de la zona en la distribución en las grietas que se
forman en términos o (1) la mayor distancia euclideana entre la grieta
 y el exterior de la pieza o (2) sí o no parte la pieza.
'''

import seaborn as sns
import matplotlib.pyplot as plt
from math import fabs, sqrt
from scipy.stats import describe
from PIL import Image, ImageColor
from random import randint, choice





def celda(pos):
    if pos in semillas:
        return semillas.index(pos)
    x, y = pos % n, pos // n
    cercano = None
    menor = n * sqrt(2)
    for i in range(k):
        (xs, ys) = semillas[i]
        dx, dy = x - xs, y - ys
        dist = sqrt(dx**2 + dy**2)
        if dist < menor:
            cercano, menor = i, dist
    return cercano

def inicio():
    direccion = randint(0, 3)
    if direccion == 0: # vertical abajo -> arriba
        return (0, randint(0, n - 1))
    elif direccion == 1: # izq. -> der
        return (randint(0, n - 1), 0)
    elif direccion == 2: # der. -> izq.
        return (randint(0, n - 1), n - 1)
    else:
        return (n - 1, randint(0, n - 1))



def propagar(replica):
    xcent, ycent, dist = 0, 0, 0
    prob, dificil = 0.9, 0.8
    mitad = n / 2
    mindist = sqrt(((1)**2) + ((1)**2))
    grieta = voronoi.copy()
    g = grieta.load()
    (x, y) = inicio()
    largo = 0
    negro = (0, 0, 0)
    while True:
        g[x, y] = negro
        largo += 1
        frontera, interior = [], []
        for v in vecinos:
            (dx, dy) = v
            vx, vy = x + dx, y + dy
            if vx >= 0 and vx < n and vy >= 0 and vy < n: # existe
               if g[vx, vy] != negro: # no tiene grieta por el momento
                   if vor[vx, vy] == vor[x, y]: # misma celda
                       interior.append(v)
                   else:
                       frontera.append(v)
        elegido = None
        #--------------- Mayor Distancia por cuadrante ---------------

        if x < mitad and y < mitad:
            xout = 0
            yout = 0
            dist = sqrt(((x - xout) **2) + ((y - yout)**2))
        elif x > mitad and y < mitad:
            xout = n
            yout = 0
            dist = sqrt(((xout - x)**2) + ((yout + y)**2))
        elif x < mitad and y > mitad:
            xout = 0
            yout = n
            dist = sqrt(((xout + x)**2) + ((yout - y)**2))
        elif x > mitad and y > mitad:
            xout = n
            yout = n
            dist = sqrt(((xout - x)**2) + ((yout - y)**2))



        if dist > mindist:
            mindist = dist
            maxdist= dist


        #--------------- Mayor Distancia por cuadrante ---------------
        if len(frontera) > 0:
            elegido = choice(frontera)
            prob = 1
        elif len(interior) > 0:
            elegido = choice(interior)
            prob *= dificil
        if elegido is not None:
            (dx, dy) = elegido
            x, y = x + dx, y + dy
        else:
            break # ya no se propaga

    return maxdist






N = [50, 100, 150, 200]
K = [1, 2, 4, 8, 16, 32]
gg = []
for n in N:
    print("------------------",n ,"------------------")
    gg = []
    for o in K:
        print("------------------",n,o,"------------------")
        semillas = []
        k = n * o
        for s in range(k):
            while True:
                x, y = randint(0, n - 1), randint(0, n - 1)
                if (x, y) not in semillas:
                    semillas.append((x, y))
                    break

        mitad = n / 2
        mindist = sqrt(((mitad)**2) + ((mitad)**2))


        celdas = [celda(i) for i in range(n * n)]
        voronoi = Image.new('RGB', (n, n))
        vor = voronoi.load()
        c = sns.color_palette("Set3", k).as_hex()
        for i in range(n * n):
            vor[i % n, i // n] = ImageColor.getrgb(c[celdas.pop(0)])
            limite, vecinos = n, []
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if dx != 0 or dy != 0:
                        vecinos.append((dx, dy))



        g = []
        for r in range(100): # pruebas sin paralelismo
            g.append(propagar(r))

        gg.append(g)
        #print(g)
        print("------------------",n,o,"------------------")
    print("------------------",n ,"------------------")
    if n == 50:
        gg1=gg
    elif n == 100:
        gg2=gg
    elif n == 150:
        gg3=gg
    elif n == 200:
        gg4=gg

print("")
print(gg1)
print("")
print("")
print(gg2)
print("")
print("")
print(gg3)
print("")
print("")
print(gg4)
print("")

plt.subplot(221)
plt.boxplot(gg1)
plt.xticks([1, 2, 3, 4, 5, 6], ['1', '2', '4', '8', '16', '32'])
plt.xlabel('Coeficiente de semillas')
plt.ylabel(' Mayor distancia euclideana')
plt.title('a) 50×50')


plt.subplot(222)
plt.boxplot(gg2)
plt.xticks([1, 2, 3, 4, 5, 6], ['1', '2', '4', '8', '16', '32'])
plt.xlabel('Coeficiente de semillas')
plt.ylabel(' Mayor distancia euclideana')
plt.title('b) 100×100')


plt.subplot(223)
plt.boxplot(gg3)
plt.xticks([1, 2, 3, 4, 5, 6], ['1', '2', '4', '8', '16', '32'])
plt.xlabel('Coeficiente de semillas')
plt.ylabel(' Mayor distancia euclideana')
plt.title('c) 150×150')


plt.subplot(224)
plt.boxplot(gg4)
plt.xticks([1, 2, 3, 4, 5, 6], ['1', '2', '4', '8', '16', '32'])
plt.xlabel('Coeficiente de semillas')
plt.ylabel(' Mayor distancia euclideana')
plt.title('d) 200×200')


plt.subplots_adjust(top=0.95, bottom=0.08, left=0.05, right=0.95, hspace=0.35,
                        wspace=0.2)

plt.show()
plt.close()
