'''                             TAREA9
Agrega a cada partícula una masa y haz que la masa cause fuerzas
gravitacionales (atracciones) además de las fuerzas causadas por
las cargas. Estudia la distribución de velocidades de las partículas
y verifica gráficamente que esté presente una relación entre los tres
factores: la velocidad, la magnitud de la carga, y la masa de las
partículas. Toma en cuenta que la velocidad también es afectada por
las posiciones.
'''
# -*- coding: utf-8 -*
import numpy as np
import pandas as pd

G = 6.674*10**(-11) #constante de gravitacion universal


n = 50
x = np.random.normal(size = n)
y = np.random.normal(size = n)
c = np.random.normal(size = n)
m = np.random.normal(size = n)
xmax = max(x)
xmin = min(x)
x = (x - xmin) / (xmax - xmin) # de 0 a 1
ymax = max(y)
ymin = min(y)
y = (y - ymin) / (ymax - ymin)
cmax = max(c)
cmin = min(c)
c = 2 * (c - cmin) / (cmax - cmin) - 1 # entre -1 y 1
g = np.round(5 * c).astype(int)
mmax = max(m)
mmin = min(m)
m = (10 * (m - mmin) / (mmax - mmin) - 1)  # entre 0 y 10
m = np.round(m).astype(int)
m = abs(m * 10)
p = pd.DataFrame({'x': x, 'y': y, 'c': c, 'g': g, 'm': m})
paso = 256 // 10
niveles = [i/256 for i in range(0, 256, paso)]
colores = [(niveles[i], 0, niveles[-(i + 1)]) for i in range(len(niveles))]

x2 = x
y2 = y
c2 = c
g2 = g
p2 = pd.DataFrame({'x2': x2, 'y2': y2, 'c2': c2, 'g2': g2})
paso2 = paso
niveles2 = niveles
colores2 = colores




import matplotlib.pyplot as plt
import matplotlib.colorbar as colorbar
from matplotlib.colors import LinearSegmentedColormap

palette = LinearSegmentedColormap.from_list('tonos', colores, N = len(colores))
palette2 = LinearSegmentedColormap.from_list('tonos', colores2, N = len(colores2))


from math import fabs, sqrt, floor, log
eps = 0.001

def fuerza(i):
    pi = p.iloc[i]
    xi = pi.x
    yi = pi.y
    ci = pi.c
    fx, fy = 0, 0
    for j in range(n):
        pj = p.iloc[j]
        xj = pj.x
        yj = pj.y
        cj = pj.c
        dire = (-1)**(1 + (ci * cj < 0))
        dx = xi - pj.x
        dy = yi - pj.y
        factor = dire * fabs(ci - cj) / (sqrt(dx**2 + dy**2) + eps)
        fx -= dx * factor
        fy -= dy * factor
    return (fx, fy)

def fuerza2(i):
    pi = p2.iloc[i]
    xi = pi.x2
    yi = pi.y2
    ci = pi.c2
    fx, fy = 0, 0
    for j in range(n):
        pj = p2.iloc[j]
        xj = pj.x2
        yj = pj.y2
        cj = pj.c2
        dire = (-1)**(1 + (ci * cj < 0))
        dx = xi - pj.x2
        dy = yi - pj.y2
        factor = dire * fabs(ci - cj) / (sqrt(dx**2 + dy**2) + eps)
        fx -= dx * factor
        fy -= dy * factor
    return (fx, fy)

def fuerzaG(i):
    pi = p.iloc[i]
    xi = pi.x
    yi = pi.y
    mi = pi.m
    fgx, fgy = 0, 0
    for j in range(n):
        pj = p.iloc[j]
        xj = pj.x
        yj = pj.y
        mj = pj.m
        dx = xi - pj.x
        dy = yi - pj.y
        factor = (G * ((mi * mj) / (sqrt((dx**2) + (dy**2)) + eps)**2))
        fgx = dx * factor
        fgy = dy * factor
    return(fgx*10000000, fgy*10000000)





tmax = 150
digitos = floor(log(tmax, 10)) + 1
fig, ax = plt.subplots(figsize=(7.5, 6.5), ncols=1)
pos = plt.scatter(p.x, p.y, c = p.g, s = m, cmap = palette)
handles, labels = pos.legend_elements(prop="sizes", alpha=0.5)
legend2 = ax.legend(handles, labels, loc="upper right", title="Masa")
fig.colorbar(pos, ax=ax)
plt.title('Estado inicial')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
fig.savefig('p9p_t0.png')


def actualiza(pos, fuerza, de):
    return max(min(pos + de * fuerza, 1), 0)

def suma(f, fg):
    sumax = []
    sumay = []
    for v in range(n):
        FX= f[v][0]
        FGX= fg[v][0]
        sumax.append(FX + FGX)

        FY = f[v][1]
        FGY = fg[v][1]
        sumay.append(FY + FGY)

    return list(zip(sumax,sumay))



import multiprocessing
from itertools import repeat


if __name__ == "__main__":
    for t in range(tmax):
        print("-----",t ,"-----")

        f = []
        fg = []
        Actx = []
        Acty = []
        q = 0
        Q = 0
        for i in range(n):
            f.append(fuerza(i))
            fg.append(fuerzaG(i))

        F = suma(f, fg)


        delta = 0.02 / max([max(fabs(fx), fabs(fy)) for (fx, fy) in F])


        for v in F:
            Actx.append(actualiza(p.x[q],v[0],delta))
            q = q + 1
        p['x'] = Actx

        for v in F:
            Acty.append(actualiza(p.y[Q],v[1],delta))
            Q = Q + 1
        p['y'] = Acty


        fig, ax = plt.subplots(figsize=(7.5, 6.5), ncols=1)
        pos = plt.scatter(p.x, p.y, c = p.g, s = m, cmap = palette)
        handles, labels = pos.legend_elements(prop="sizes", alpha=0.5)
        legend2 = ax.legend(handles, labels, loc="upper right", title="Masa")
        fig.colorbar(pos, ax=ax)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.title('Paso {:d}'.format(t + 1))
        fig.savefig('p9p_t' + format(t + 1, '0{:d}'.format(digitos)) + '.png')
        print("-----",t ,"-----")

    print()

    digitos2 = floor(log(tmax, 10)) + 1
    fig2, ax2 = plt.subplots(figsize=(7.5, 6.5), ncols=1)
    pos2 = plt.scatter(p2.x2, p2.y2, c = p2.g2, cmap = palette2)
    handles2, labels2 = pos2.legend_elements(prop="sizes", alpha=0.5)
    fig2.colorbar(pos2, ax=ax2)
    plt.title('Estado inicial')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    fig2.savefig('p9p_2t0.png')

    for t in range(tmax):
        print("-----",t ,"-----")

        f2 = []
        Actx2 = []
        Acty2 = []
        q = 0
        Q = 0
        for i in range(n):
            f2.append(fuerza2(i))

        delta2 = 0.02 / max([max(fabs(fx2), fabs(fy2)) for (fx2, fy2) in f2])

        for v in f2:
            Actx2.append(actualiza(p2.x2[q],v[0],delta2))
            q = q + 1
        p2['x2'] = Actx2

        for v in f2:
            Acty2.append(actualiza(p2.y2[Q],v[1],delta2))
            Q = Q + 1
        p2['y2'] = Acty2

        fig2, ax2 = plt.subplots(figsize=(7.5, 6.5), ncols=1)
        pos2 = plt.scatter(p2.x2, p2.y2, c = p2.g2, cmap = palette2)
        handles2, labels2 = pos2.legend_elements(prop="sizes", alpha=0.5)
        fig2.colorbar(pos2, ax=ax2)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.title('Paso {:d}'.format(t + 1))
        fig2.savefig('p9p_2t' + format(t + 1, '0{:d}'.format(digitos2)) + '.png')
        plt.close()
        print("-----",t ,"-----")
