'''                             TAREA10
En este código se demuestra graficamente la diferencia entre
la simulacion con y sin ruleta.
'''

# -*- coding: utf-8 -*
import numpy as np
import pandas as pd
from time import time
from random import choices
from random import random, randint, sample

def knapsack(peso_permitido, pesos, valores):
    assert len(pesos) == len(valores)
    peso_total = sum(pesos)
    valor_total = sum(valores)
    if peso_total < peso_permitido:
        return valor_total
    else:
        V = dict()
        for w in range(peso_permitido + 1):
            V[(w, 0)] = 0
        for i in range(len(pesos)):
            peso = pesos[i]
            valor = valores[i]
            for w in range(peso_permitido + 1):
                cand = V.get((w - peso, i), -float('inf')) + valor
                V[(w, i + 1)] = max(V[(w, i)], cand)
        return max(V.values())

def factible(seleccion, pesos, capacidad):
    return np.inner(seleccion, pesos) <= capacidad

def objetivo(seleccion, valores):
    return np.inner(seleccion, valores)

def normalizar(data):
    menor = min(data)
    mayor = max(data)
    rango  = mayor - menor
    data = data - menor # > 0
    return data / rango # entre 0 y 1

def generador_pesos(cuantos, low, high):
    return np.round(normalizar(np.random.normal(size = cuantos)) * (high - low) + low)

def generador_valores(pesos, low, high):
    n = len(pesos)
    valores = np.empty((n))
    for i in range(n):
        valores[i] = np.random.normal(pesos[i], random(), 1)
    return normalizar(valores) * (high - low) + low

def poblacion_inicial(n, tam):
    pobl = np.zeros((tam, n))
    for i in range(tam):
        pobl[i] = (np.round(np.random.uniform(size = n))).astype(int)
    return pobl

def mutacion(sol, n):
    pos = randint(0, n - 1)
    mut = np.copy(sol)
    mut[pos] = 1 if sol[pos] == 0 else 0
    return mut

def reproduccion(x, y, n):
    pos = randint(2, n - 2)
    xy = np.concatenate([x[:pos], y[pos:]])
    yx = np.concatenate([y[:pos], x[pos:]])
    return (xy, yx)

def fitness(fact,obj):
    if fact != True:
        obj = obj * .1     # 10% de obj
    return obj

def ruleta(poblacion, fitnes):
    sel = {}
    while len(sel) < 2:
        sel = set(choices(poblacion, weights = fitnes, k = 2))
    return list(sel)

n = 50    #numero de objetos
pesos = generador_pesos(n, 15, 80)
valores = generador_valores(pesos, 10, 500)
capacidad = int(round(sum(pesos) * 0.65))
Toptimo1 = time()
optimo = knapsack(capacidad, pesos, valores)
Toptimo2 = time()
print("El tiempo optimo tardo:",Toptimo2 - Toptimo1,"seg")
init = 200  #numero de 0 o 1
p = poblacion_inicial(n, init)
#print(p)
tam = p.shape[0]
#print(tam)
#print()
assert tam == init
pm = 0.05       # probabilidad a mutar
rep = 50  #50    parajas a reporducirse
tmax = 50  #50    iteraciones
mejor = None
mejores = []

#---------- Variables2 ----------
pesos2 = pesos
valores2 = valores
capacidad2 = capacidad
optimo2 = knapsack(capacidad2, pesos2, valores2)
init2 = 200  #numero de 0 o 1
p2 = poblacion_inicial(n, init2)
#print(p)
tam2 = p2.shape[0]
#print(tam)
#print()
assert tam2 == init2
mejor2 = None
mejores2 = []
#---------- Variables2 ----------






for t in range(tmax):
    for i in range(tam): # mutarse con probabilidad pm
        if random() < pm:
            p = np.vstack([p, mutacion(p[i], n)])
    for i in range(rep):  # reproducciones
        padres = sample(range(tam), 2)
        hijos = reproduccion(p[padres[0]], p[padres[1]], n)
        p = np.vstack([p, hijos[0], hijos[1]])
    tam = p.shape[0]
    d = []
    for i in range(tam):
        d.append({'idx': i, 'obj': objetivo(p[i], valores),
                  'fact': factible(p[i], pesos, capacidad)})
    d = pd.DataFrame(d).sort_values(by = ['fact', 'obj'], ascending = False)
    mantener = np.array(d.idx[:init])
    p = p[mantener, :]
    tam = p.shape[0]
    assert tam == init
    factibles = d.loc[d.fact == True,]
    mejor = max(factibles.obj)
    mejores.append(mejor)









for t in range(tmax):

#---------- Generar fit ----------
    fit = []
    fitobj = []
    fitfac = []
    tam2 = p2.shape[0]
    for i in range(tam2):
        fitobj.append(objetivo(p2[i], valores2))
        fitfac.append(factible(p2[i], pesos2, capacidad2))
    FIT = list(zip(fitobj,fitfac))

    #print(FIT)

    for i in range(len(FIT)):
        fit.append(fitness(FIT[i][1],FIT[i][0]))
    #print()
    #print(tam)
    #print()
    #print(fit)
    #print()
#---------- Generar fit ----------


#---------- Mutaciones ----------
    for i in range(tam2): # mutarse con probabilidad pm
        if random() < pm:
            p2 = np.vstack([p2, mutacion(p2[i], n)])
#---------- Mutaciones ----------
#---------- Reproducciones ----------
    for i in range(rep):  # reproducciones
        padres2 = ruleta(range(len(fitobj)), fit)
        hijos2 = reproduccion(p2[padres2[0]], p2[padres2[1]], n)
        p2 = np.vstack([p2, hijos2[0], hijos2[1]])
#---------- Reproducciones ----------

    tam2 = p2.shape[0]      # tam es el numero de filas en la matriz
    d2 = []
    for i in range(tam2):
        d2.append({'idx': i, 'obj': objetivo(p2[i], valores2),
                  'fact': factible(p2[i], pesos2, capacidad2)})
    d2 = pd.DataFrame(d2).sort_values(by = ['fact', 'obj'], ascending = False)
    mantener2 = np.array(d2.idx[:init])
    p2 = p2[mantener2, :]
    tam2 = p2.shape[0]
    assert tam2 == init
    factibles2 = d2.loc[d2.fact == True,]
    mejor2 = max(factibles2.obj)
    mejores2.append(mejor2)

import matplotlib.pyplot as plt


plt.figure(figsize=(7, 3))
plt.axhline(y = optimo2, color = 'green', label = 'Óptimo')
plt.plot(range(tmax), mejores,linewidth=1, markersize=3, label = 'Sin ruleta', color = 'blue')
plt.plot(range(tmax), mejores2, linewidth=1, markersize=3, label = 'Con ruleta', color = 'red')
plt.xlabel('Paso')
plt.ylabel('Mayor valor')
plt.legend()
plt.ylim(0.95 * min(mejores2), 1.05 * optimo2)
plt.show()
plt.close()
print(mejor2, (optimo2 - mejor2) / optimo2)
