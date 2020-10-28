'''                             TAREA6
Vacuna con probabilidad pv a los agentes al momento de crearlos de
tal forma que están desde el inicio en el estado R y ya no podrán
contagiarse ni propagar la infección. Estudia el efecto estadístico
del valor de pv en (de cero a uno en pasos de 0.1) el porcentaje
máximo de infectados durante la simulación y el momento (iteración)
en el cual se alcanza ese máximo.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import floor, log, sqrt
from random import random, uniform

l = 1.5
n = 100          # Número de agentes 
pi = 0.05       # Probabilidad de contagio inicial
pr = 0.02       # prob. de recuperar
pv = [d for d in np.arange(0, 1, 0.1)]       # prob. de vacuna
v = l / 30      # Velocidad
r = 0.1         # distancia segura
tmax = 200      #tiempo maximo de simulación
digitos = floor(log(tmax, 10)) + 1
c = {'I': 'r', 'S': 'g', 'R': 'orange'}
m = {'I': 'o', 'S': 's', 'R': '2'}
conjuntox_epidemias=[]
conjuntoy_epidemias=[]



for pv in pv:
    print("-----------------", pv, "-----------------")

#----------------- Creación de DataFrame -----------------
    agentes =  pd.DataFrame()
    agentes['x'] = [uniform(0, l) for i in range(n)]
    agentes['y'] = [uniform(0, l) for i in range(n)]
    agentes['dx'] = [uniform(-v, v) for i in range(n)]
    agentes['dy'] = [uniform(-v, v) for i in range(n)]

    for i in range(n):
        agentes.at[i, 'estado'] = 'S'
        if random() < pi:
            agentes.at[i, 'estado'] = 'I'
        else:
            if random() < pv:
                agentes.at[i, 'estado'] = 'R'


    epidemia = []
#----------------- Creación de DataFrame -----------------


#----------------- Conteo de los Infactados -----------------
    for tiempo in range(tmax):
        conteos = agentes.estado.value_counts()     #conteo de los dif estados
        infectados = conteos.get('I', 0)            # de conteos obtengo solo los infectados
        epidemia.append(infectados)                 # historial de infectados
        if infectados == 0:             # si no hay infectados se termina la simulacion
            break
#----------------- Conteo de los infectados -----------------


#----------------- Evaluando posibles Contagios -----------------
        contagios = [False for i in range(n)]
        for i in range(n):                  # Probando para cada agente
            a1 = agentes.iloc[i]            # Datos del Agente evaluando
            if a1.estado == 'I':            # Si el evaluando es Infectado
                for j in range(n):          # Evaluas posible infeccción para todos los demas agentes
                    a2 = agentes.iloc[j]    # Datos del Agente evaluando
                    if a2.estado == 'S':
                        d = sqrt((a1.x - a2.x)**2 + (a1.y - a2.y)**2)  #distancia entre agentes
                        if d < r:
                            if random() < (r - d) / r:
                                contagios[j] = True
#----------------- Evaluando posibles Contagios -----------------

#-------------- Actualizano Contagios, Recuperados y Movimientos --------------
        for i in range(n):                  # Probando para cada agente
            a = agentes.iloc[i]             # Datos del Agente evaluando
            if contagios[i]:                # Si evaluando a sido infectado
                agentes.at[i, 'estado'] = 'I'   # Actualiza su estado
            elif a.estado == 'I':           # Si evaluando es infectado
                if random() < pr:           # Si Pr es mayor
                    agentes.at[i, 'estado'] = 'R'   # Evaluando se recupera
            x = a.x + a.dx
            y = a.y + a.dy
            x = x if x < l else x - l
            y = y if y < l else y - l
            x = x if x > 0 else x + l
            y = y if y > 0 else y + l
            agentes.at[i, 'x'] = x          # Actualizando posicion en X
            agentes.at[i, 'y'] = y          # Actualizando posicion en X
#-------------- Actualizano Contagios y Movimientos --------------

#-------- Calculo conagio .max y tiempo de contagio .max --------
    cont = 0
    max_contagio = max(epidemia)
    for x in range(len(epidemia)):
        if epidemia[x] == max_contagio:
            cont = cont + 1
            if cont == 1:
                momento_max_contagio = x

    print("Máximo contagio:", max_contagio)
    print("El máximo contagio se tuvo en la iteración:", momento_max_contagio )
#-------- Calculo conagio .max y tiempo de contagio .max --------



    conjuntox_epidemias.append(epidemia)
    conjuntoy = [100 * e / n for e in epidemia]
    conjuntoy_epidemias. append(conjuntoy)

    print("-----------------", pv, "-----------------")


#----------------- Graficando -----------------
label=['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9']
for a in range(len(conjuntox_epidemias)):
    plt.plot(range(len(conjuntox_epidemias[a])),conjuntoy_epidemias[a], label=label[a])
plt.xlabel('Tiempo')
plt.ylabel('Porcentaje de infectados (%)')
plt.legend()
plt.show()
plt.close()
#----------------- Graficando -----------------
