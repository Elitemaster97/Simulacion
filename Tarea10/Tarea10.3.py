'''                             TAREA10                           '''



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

def simu(n,k):
    pesos = generador_pesos(n, 15, 80)
    valores = generador_valores(pesos, 10, 500)
    capacidad = int(round(sum(pesos) * 0.65))

    gg = 0
    Toptimo1 = time()
    optimo = knapsack(capacidad, pesos, valores)
    Toptimo2 = time()
    TiempoO = Toptimo2 - Toptimo1
    X = optimo / TiempoO


    init = 50  #numero de 0 o 1
    p = poblacion_inicial(n, init)
    tam = p.shape[0]

    assert tam == init
    pm = 0.05       # probabilidad a mutar
    rep = round(n * .2)  #50    parajas a reporducirse
    tmax = 100  #50    iteraciones
    mejor = None
    mejores = []


    cont = 0
    TTin = []
    Tres1 = time()
    for t in range(tmax):
        cont = cont + 1
        Tin1 = time()

        if k == 0:
            #---------- Generar fit ----------
            fit = []
            fitobj = []
            fitfac = []
            tam = p.shape[0]
            for i in range(tam):
                fitobj.append(objetivo(p[i], valores))
                fitfac.append(factible(p[i], pesos, capacidad))
            FIT = list(zip(fitobj,fitfac))


            for i in range(len(FIT)):
                fit.append(fitness(FIT[i][1],FIT[i][0]))

            #---------- Generar fit ----------


            #---------- Mutaciones ----------
            for i in range(tam): # mutarse con probabilidad pm
                if random() < pm:
                    p = np.vstack([p, mutacion(p[i], n)])
            #---------- Mutaciones ----------
            #---------- Reproducciones ----------
            for i in range(rep):  # reproducciones
                padres = ruleta(range(len(fitobj)), fit)
                hijos = reproduccion(p[padres[0]], p[padres[1]], n)
                p = np.vstack([p, hijos[0], hijos[1]])
            #---------- Reproducciones ----------

        if k == 1:
            for i in range(tam): # mutarse con probabilidad pm
                if random() < pm:
                    p = np.vstack([p, mutacion(p[i], n)])
            for i in range(rep):  # reproducciones
                padres = sample(range(tam), 2)
                hijos = reproduccion(p[padres[0]], p[padres[1]], n)
                p = np.vstack([p, hijos[0], hijos[1]])


        tam = p.shape[0]      # tam es el numero de filas en la matriz
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

        Tin2 = time()
        TTin.append(Tin2 - Tin1)
        TTTin = sum(TTin)

        Y = mejor/TTTin

        if TTTin >= TiempoO:
            break

        if Y < X :
            gg = 123
            return gg, n

        if tmax == cont:
            o=1

    Tres2 = time()




def simu2(n,k):
    pesos = generador_pesos2(n, 15, 80)
    valores = generador_valores2(pesos, 10, 500)
    capacidad = int(round(sum(pesos) * 0.65))

    gg = 0
    Toptimo1 = time()
    optimo = knapsack(capacidad, pesos, valores)
    Toptimo2 = time()
    TiempoO = Toptimo2 - Toptimo1
    X = optimo / TiempoO


    init = 50  #numero de 0 o 1
    p = poblacion_inicial(n, init)
    tam = p.shape[0]

    assert tam == init
    pm = 0.05       # probabilidad a mutar
    rep = round(n * .2)  #50    parajas a reporducirse
    tmax = 100  #50    iteraciones
    mejor = None
    mejores = []


    cont = 0
    TTin = []
    Tres1 = time()
    for t in range(tmax):
        cont = cont + 1
        Tin1 = time()

        if k == 0:
            #---------- Generar fit ----------
            fit = []
            fitobj = []
            fitfac = []
            tam = p.shape[0]
            for i in range(tam):
                fitobj.append(objetivo(p[i], valores))
                fitfac.append(factible(p[i], pesos, capacidad))
            FIT = list(zip(fitobj,fitfac))


            for i in range(len(FIT)):
                fit.append(fitness(FIT[i][1],FIT[i][0]))

            #---------- Generar fit ----------


            #---------- Mutaciones ----------
            for i in range(tam): # mutarse con probabilidad pm
                if random() < pm:
                    p = np.vstack([p, mutacion(p[i], n)])
            #---------- Mutaciones ----------
            #---------- Reproducciones ----------
            for i in range(rep):  # reproducciones
                padres = ruleta(range(len(fitobj)), fit)
                hijos = reproduccion(p[padres[0]], p[padres[1]], n)
                p = np.vstack([p, hijos[0], hijos[1]])
            #---------- Reproducciones ----------

        if k == 1:
            for i in range(tam): # mutarse con probabilidad pm
                if random() < pm:
                    p = np.vstack([p, mutacion(p[i], n)])
            for i in range(rep):  # reproducciones
                padres = sample(range(tam), 2)
                hijos = reproduccion(p[padres[0]], p[padres[1]], n)
                p = np.vstack([p, hijos[0], hijos[1]])

        tam = p.shape[0]      # tam es el numero de filas en la matriz
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

        Tin2 = time()
        TTin.append(Tin2 - Tin1)
        TTTin = sum(TTin)

        Y = mejor/TTTin

        if TTTin >= TiempoO:
            break

        if Y < X :
            gg = 123
            return gg, n

        if tmax == cont:
            o=1

    Tres2 = time()



def simu3(n,k):
    pesos = generador_pesos3(n, 15, 80)
    valores = generador_valores3(pesos)
    capacidad = int(round(sum(pesos) * 0.65))

    gg = 0
    Toptimo1 = time()
    optimo = knapsack(capacidad, pesos, valores)
    Toptimo2 = time()
    TiempoO = Toptimo2 - Toptimo1
    X = optimo / TiempoO


    init = 50  #numero de 0 o 1
    p = poblacion_inicial(n, init)
    tam = p.shape[0]

    assert tam == init
    pm = 0.05       # probabilidad a mutar
    rep = round(n * .2)  #50    parajas a reporducirse
    tmax = 100  #50    iteraciones
    mejor = None
    mejores = []


    cont = 0
    TTin = []
    Tres1 = time()
    for t in range(tmax):
        cont = cont + 1
        Tin1 = time()

        if k == 0:
            #---------- Generar fit ----------
            fit = []
            fitobj = []
            fitfac = []
            tam = p.shape[0]
            for i in range(tam):
                fitobj.append(objetivo(p[i], valores))
                fitfac.append(factible(p[i], pesos, capacidad))
            FIT = list(zip(fitobj,fitfac))


            for i in range(len(FIT)):
                fit.append(fitness(FIT[i][1],FIT[i][0]))

            #---------- Generar fit ----------


            #---------- Mutaciones ----------
            for i in range(tam): # mutarse con probabilidad pm
                if random() < pm:
                    p = np.vstack([p, mutacion(p[i], n)])
            #---------- Mutaciones ----------
            #---------- Reproducciones ----------
            for i in range(rep):  # reproducciones
                padres = ruleta(range(len(fitobj)), fit)
                hijos = reproduccion(p[padres[0]], p[padres[1]], n)
                p = np.vstack([p, hijos[0], hijos[1]])
            #---------- Reproducciones ----------

        if k == 1:
            for i in range(tam): # mutarse con probabilidad pm
                if random() < pm:
                    p = np.vstack([p, mutacion(p[i], n)])
            for i in range(rep):  # reproducciones
                padres = sample(range(tam), 2)
                hijos = reproduccion(p[padres[0]], p[padres[1]], n)
                p = np.vstack([p, hijos[0], hijos[1]])

        tam = p.shape[0]      # tam es el numero de filas en la matriz
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

        Tin2 = time()
        TTin.append(Tin2 - Tin1)
        TTTin = sum(TTin)

        Y = mejor/TTTin

        if TTTin >= TiempoO:
            break

        if Y < X :
            gg = 123
            return gg, n

        if tmax == cont:
            o=1

    Tres2 = time()



def simu4(n,k):
    pesos = generador_pesos3(n, 15, 80)
    valores = generador_valores3(pesos)
    capacidad = int(round(sum(pesos) * 0.65))

    gg = 0
    Toptimo1 = time()
    optimo = knapsack(capacidad, pesos, valores)
    Toptimo2 = time()
    TiempoO = Toptimo2 - Toptimo1
    X = optimo / TiempoO


    init = 50  #numero de 0 o 1
    p = poblacion_inicial(n, init)
    tam = p.shape[0]

    assert tam == init
    pm = 0.05       # probabilidad a mutar
    rep = round(n * .2)  #50    parajas a reporducirse
    tmax = 100  #50    iteraciones
    mejor = None
    mejores = []


    cont = 0
    TTin = []
    Tres1 = time()
    for t in range(tmax):
        cont = cont + 1
        Tin1 = time()

        if k == 0:
            #---------- Generar fit ----------
            fit = []
            fitobj = []
            fitfac = []
            tam = p.shape[0]
            for i in range(tam):
                fitobj.append(objetivo(p[i], valores))
                fitfac.append(factible(p[i], pesos, capacidad))
            FIT = list(zip(fitobj,fitfac))


            for i in range(len(FIT)):
                fit.append(fitness(FIT[i][1],FIT[i][0]))

            #---------- Generar fit ----------


            #---------- Mutaciones ----------
            for i in range(tam): # mutarse con probabilidad pm
                if random() < pm:
                    p = np.vstack([p, mutacion(p[i], n)])
            #---------- Mutaciones ----------
            #---------- Reproducciones ----------
            for i in range(rep):  # reproducciones
                padres = ruleta(range(len(fitobj)), fit)
                hijos = reproduccion(p[padres[0]], p[padres[1]], n)
                p = np.vstack([p, hijos[0], hijos[1]])
            #---------- Reproducciones ----------

        if k == 1:
            for i in range(tam): # mutarse con probabilidad pm
                if random() < pm:
                    p = np.vstack([p, mutacion(p[i], n)])
            for i in range(rep):  # reproducciones
                padres = sample(range(tam), 2)
                hijos = reproduccion(p[padres[0]], p[padres[1]], n)
                p = np.vstack([p, hijos[0], hijos[1]])

        tam = p.shape[0]      # tam es el numero de filas en la matriz
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

        Tin2 = time()
        TTin.append(Tin2 - Tin1)
        TTTin = sum(TTin)

        Y = mejor/TTTin

        if TTTin >= TiempoO:
            break

        if Y < X :
            gg = 123
            return gg, n

        if tmax == cont:
            o=1

    Tres2 = time()



def generador_pesos2(cuantos, low, high):
    return np.round(normalizar(np.random.exponential(size = cuantos)) * (high - low) + low)

def generador_valores2(pesos, low, high):
    n = len(pesos)
    return np.round(normalizar(np.random.exponential(size = n)) * (high - low) + low)

def generador_pesos3(cuantos, low, high):
    return np.round(normalizar(np.random.exponential(size = cuantos)) * (high - low) + low)

def generador_valores3(pesos):
    VyR = []
    n = len(pesos)
    valores = np.round(pesos * 300)
    prom = (sum(valores)/len(valores)) * 0.25
    ruido = np.round(normalizar(np.random.normal(size = n )) * (prom - (-prom)) + (-prom))

    for i in range(n):
        VyR.append(valores[i] + ruido[i])

    return VyR


def generador_pesos4(cuantos, low, high):
    return np.round(normalizar(np.random.exponential(size = cuantos)) * (high - low) + low)

def generador_valores4(pesos):
    VyR = []
    n = len(pesos)
    valores = np.round(300 / pesos)
    prom = (sum(valores)/len(valores)) * 0.25
    ruido = np.round(normalizar(np.random.normal(size = n )) * (prom - (-prom)) + (-prom))

    for i in range(n):
        VyR.append(valores[i] + ruido[i])

    return VyR

#---------- Ejecuccion ----------

expts = []
expts2 = []
expts3 = []
expts4 = []
exptsk = []
expts2k = []
expts3k = []
expts4k = []
for k in range(2):
    for l in range(4):
        for i in range(40):
            for n in range(10,100):
                if l == 0:
                    exp = []
                    exp = simu(n,k)
                    if exp != None:
                        if exp[0] == 123:
                            if k == 0:
                                expts.append(exp[1])
                                break
                            if k == 1:
                                exptsk.append(exp[1])
                                break

                if l == 1:
                    exp2 = []
                    exp2 = simu2(n,k)
                    if exp2 != None:
                        if exp2[0] == 123:
                            if k == 0:
                                expts2.append(exp2[1])
                                break
                            if k == 1:
                                expts2k.append(exp2[1])
                                break

                if l == 2:
                    exp3 = []
                    exp3 = simu3(n,k)
                    if exp3 != None:
                        if exp3[0] == 123:
                            if k == 0:
                                expts3.append(exp3[1])
                                break
                            if k == 1:
                                expts3k.append(exp3[1])
                                break

                if l == 3:
                    exp4 = []
                    exp4 = simu4(n,k)
                    if exp4 != None:
                        if exp4[0] == 123:
                            if k == 0:
                                expts4.append(exp4[1])
                                break
                            if k == 1:
                                expts4k.append(exp4[1])
                                break



EXPTS = []

EXPTS.append(expts)
EXPTS.append(exptsk)

EXPTS.append(expts2)
EXPTS.append(expts2k)

EXPTS.append(expts3)
EXPTS.append(expts3k)

EXPTS.append(expts4)
EXPTS.append(expts4k)


#---------- Ejecuccion ----------


import matplotlib.pyplot as plt
from matplotlib.patches import Polygon



nombres = ['Ejemplo','Ejemplo','Instancia 1','Instancia 1','Instancia 2','Instancia 2','Instancia 3','Instancia 3']

fig, ax1 = plt.subplots(figsize=(8, 6))
fig.canvas.set_window_title('A Boxplot Example')
plt.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

bp = plt.boxplot(EXPTS, notch=0, sym='+', vert=1, whis=1.5)

# Maya
ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)

# leyenda
ax1.set_axisbelow(True)
ax1.set_ylabel('Número de objetos (n)', fontsize=10)


numDists = 4
boxColors = ['darkkhaki', 'royalblue']
numBoxes = numDists*2
medians = list(range(numBoxes))
for i in range(numBoxes):
    box = bp['boxes'][i]
    boxX = []
    boxY = []
    for j in range(5):
        boxX.append(box.get_xdata()[j])
        boxY.append(box.get_ydata()[j])
    boxCoords = list(zip(boxX, boxY))
    # Alternanr colores
    k = i % 2
    boxPolygon = Polygon(boxCoords, facecolor=boxColors[k])
    ax1.add_patch(boxPolygon)


# ejes
xtickNames = plt.setp(ax1, xticklabels=nombres)
plt.setp(xtickNames, rotation=45, fontsize=10)

# leyendas
plt.figtext(0.83, 0.08, 'Con ruleta',
            backgroundcolor=boxColors[0], color='black', weight='roman',
            size='small',fontsize=10)
plt.figtext(0.83, 0.041, 'Sin ruleta', backgroundcolor=boxColors[1],
            color='white', weight='roman', size='small', fontsize=10)
plt.figtext(0.80, .111, '*', color='black', backgroundcolor='white',
            weight='roman', size='small', fontsize=10)
plt.figtext(0.815, 0.111, ' Simulacion:', color='black', weight='roman',
            size='small',fontsize=10)



plt.show()
