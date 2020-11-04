'''                             TAREA7
La tarea se trata de maximizar algún variante de la función
bidimensional ejemplo,g(x,y), con restricciones −3 ≤ x,y ≤ 3, con la
misma técnica del ejemplo unidimensional.Crear una visualización
animada de cómo proceden 15 réplicas simultáneas de la búsqueda
encima de una gráfica de proyección plana.
'''
import matplotlib.colorbar as colorbar
import matplotlib.pyplot as plt
from math import floor, log, e
from random import uniform
import numpy as np


def interX(x):
    return (0 + (60 - 0)) * ((x - (-3))/(3 - (-3)))

def interY(y):
    return (0 + (60 - 0)) * ((y - 2.5)/(-3.5 - 2.5))

def g(x, y):
    return ((x + 0.25)**4 - 30 * x**2 - 20 * x + (y + 0.25)**4 - 30 * y**2 - 20 * y)/50

low = -3
high = 3
LOW = 2.5
HIGH = -3.5
step = 0.1
replicas = 15

def replica(t):
    X = uniform(low, high)
    Y = uniform(LOW, HIGH)
    curr = [X,Y]
    best = curr
    for tiempo in range(t):
        paso = 0.3
        deltaX = uniform(0, paso)
        deltaY = uniform(0, paso)
        leftX = curr[0] - deltaX
        rightX = curr[0] + deltaX
        leftY = curr[1] - deltaY
        rightY = curr[1] + deltaY

        RXRY = g(rightX,rightY)
        LXRY = g(leftX,rightY)
        RXLY = g(rightX,leftY)
        LXLY = g(leftX,leftY)

        if RXRY > LXRY and RXRY > RXLY and RXRY > LXLY:
            curr = [rightX,rightY]
        if LXRY > RXRY and LXRY > RXLY and LXRY > LXLY:
            curr = [leftX,rightY]
        if RXLY > RXRY and RXLY > LXLY and RXLY > LXLY:
            curr = [rightX,leftY]
        if LXLY > RXRY and LXLY > LXRY and LXLY > RXLY:
            curr = [leftX,leftY]

        if g(curr[0],curr[1]) > g(best[0],best[1]):
            best = curr
    return best

p = np.arange(low, high, step)
n = len(p)
z = np.zeros((n, n), dtype=float)
for i in range(n):
    x = p[i]
    for j in range(n):
        y = p[n - j - 1] # voltear
        z[i, j] = g(x, y)

vgx= np.vectorize(interX)
vgy= np.vectorize(interY)

import multiprocessing
from itertools import repeat

if __name__ == "__main__":
    with multiprocessing.Pool(3) as pool:
        for q in range(0,4):

            r = pool.map(replica, repeat(10**q, replicas))
            t = range(0, n, 5)
            l = ['{:.1f}'.format(low + i * step) for i in t]
            fig, ax = plt.subplots(figsize=(6, 5), ncols=1)
            pos = ax.imshow(z)
            paraX = []
            paraY = []
            for o in range(len(r)):
                paraX.append(r[o][0])
                paraY.append(r[o][1])
            ParaX = vgx(paraX)
            ParaY = vgy(paraY)
            ax.scatter(ParaX, ParaY, marker = 'v', color = '#0000ff')
            plt.xticks(t, l)
            plt.yticks(t, l[::-1]) # arriba-abajo
            fig.colorbar(pos, ax=ax)
            plt.title('Paso {:d}'.format(10**q))
            fig.savefig('p7p_{:d}.png'.format(q), bbox_inches='tight')
            #plt.show()
            plt.close()
