'''                             TAREA8
Supongamos que cúmulos con c o más partículas (haciendo referencia al
tamaño crítico c) son suficientemente grandes para filtrar. Determina
para diversas combinaciones de k, n y el número de iteraciones t cuál
porcentaje de las partículas se logra filtrar si el filtrado se lleva
a cabo después de t iteraciones del proceso.
'''
# -*- coding: utf-8 -* # para incluir acentos
import numpy as np
from random import randint
from math import exp, floor, log

def rotura(x, c, d):
    return 1 / (1 + exp((c - x) / d))

def union(x, c):
    return exp(-x / c)

from random import random

def romperse(tam, cuantos, c, d):
    if tam == 1: # no se puede romper
        return [tam] * cuantos
    res = []
    for cumulo in range(cuantos):
        if random() < rotura(tam, c, d):
            primera = randint(1, tam - 1)
            segunda = tam - primera
            assert primera > 0
            assert segunda > 0
            assert primera + segunda == tam
            res += [primera, segunda]
        else:
            res.append(tam) # no rompió
    assert sum(res) == tam * cuantos
    return res

def unirse(tam, cuantos, c):
    res = []
    for cumulo in range(cuantos):
        if random() < union(tam, c):
            res.append(-tam) # marcamos con negativo los que quieren unirse
        else:
            res.append(tam)
    return res

def fil(k, n, duracion):
    orig = np.random.normal(size = k)
    cumulos = orig - min(orig)
    cumulos += 1 # ahora el menor vale uno
    cumulos = cumulos / sum(cumulos) # ahora suman a uno
    cumulos *= n # ahora suman a n, pero son valores decimales
    cumulos = np.round(cumulos).astype(int) # ahora son enteros
    diferencia = n - sum(cumulos) # por cuanto le hemos fallado
    cambio = 1 if diferencia > 0 else -1
    while diferencia != 0:
        p = randint(0, k - 1)
        if cambio > 0 or (cambio < 0 and cumulos[p] > 0): # sin vaciar
            cumulos[p] += cambio
            diferencia -= cambio
    assert all(cumulos != 0)
    assert sum(cumulos) == n

    c = np.median(cumulos) # tamaño crítico de cúmulos
    d = np.std(cumulos) / 4 # factor arbitrario para suavizar la curva


    for paso in range(duracion):

        assert all([c > 0 for c in cumulos])
        (tams, freqs) = np.unique(cumulos, return_counts = True)
        cumulos = []
        assert len(tams) == len(freqs)
        for i in range(len(tams)):
            cumulos += romperse(tams[i], freqs[i], c, d)

        assert all([c > 0 for c in cumulos])
        (tams, freqs) = np.unique(cumulos, return_counts = True)
        cumulos = []
        assert len(tams) == len(freqs)
        for i in range(len(tams)):
            cumulos += unirse(tams[i], freqs[i], c)
        cumulos = np.asarray(cumulos)
        neg = cumulos < 0
        a = len(cumulos)
        juntarse = -1 * np.extract(neg, cumulos) # sacarlos y hacerlos positivos
        cumulos = np.extract(~neg, cumulos).tolist() # los demás van en una lista
        assert a == len(juntarse) + len(cumulos)
        nt = len(juntarse)
        if nt > 1:
            shuffle(juntarse) # orden aleatorio
        j = juntarse.tolist()
        while len(j) > 1: # agregamos los pares formados
            cumulos.append(j.pop(0) + j.pop(0))
        if len(j) > 0: # impar
            cumulos.append(j.pop(0)) # el ultimo no alcanzó pareja
        assert len(j) == 0
        assert all([c != 0 for c in cumulos])



        if paso == duracion-1:

            total = len(cumulos)
            filtrados = []
            nofiltrados = []
            for x in range(len(cumulos)):
                if cumulos[x] > c:
                    filtrados.append(cumulos[x])
                else:
                    nofiltrados.append(cumulos[x])

            PORfiltrados = ((len(filtrados)*100)/total)
            cumulos = nofiltrados

    return PORfiltrados


from numpy.random import shuffle
import matplotlib.pyplot as plt

k = [200, 400, 600, 800]
n = [20000,40000,60000,80000]
duracion = [25, 50, 75, 100]
rep = 10

digitos = floor(log(len(k), 10)) + 1

DIM_1 = []
DIM_2 = []
DIM_3 = []
for K in k:
    DIM_3 = []
    for N in n:
        print(N)
        DIM_2 = []
        for DURACION in duracion:
            DIM_1 = []
            for i in range(rep):
                res = fil(K, N, DURACION)
                DIM_1.append(res)
            DIM_2.append(DIM_1)
        DIM_3.append(DIM_2)



    plt.subplot(221)
    plt.boxplot(DIM_3[0])
    plt.xticks([1, 2, 3, 4], duracion)
    plt.xlabel('Repetición')
    plt.ylabel('Filtrados (%)')
    plt.title('n = 20000')

    plt.subplot(222)
    plt.boxplot(DIM_3[1])
    plt.xticks([1, 2, 3, 4], duracion)
    plt.xlabel('Repetición')
    plt.ylabel('Filtrados (%)')
    plt.title('n = 40000')

    plt.subplot(223)
    plt.boxplot(DIM_3[2])
    plt.xticks([1, 2, 3, 4], duracion)
    plt.xlabel('Repetición')
    plt.ylabel('Filtrados (%)')
    plt.title('n = 60000')

    plt.subplot(224)
    plt.boxplot(DIM_3[3])
    plt.xticks([1, 2, 3, 4], duracion)
    plt.xlabel('Repetición')
    plt.ylabel('Filtrados (%)')
    plt.title('n = 80000')

    plt.subplots_adjust(top=0.95, bottom=0.08, left=0.05, right=0.95, hspace=0.35, wspace=0.2)
    plt.savefig('p8p_ct' + format(K, '0{:d}'.format(digitos)) + '.png')
    plt.close()
