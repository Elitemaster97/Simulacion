'''                             TAREA3
Examina cómo las diferencias en los tiempos de ejecución de los
diferentes ordenamientos cambian cuando se varía el número de núcleos
asignados al cluster, utilizando como datos de entrada un vector que
contiene primos grandes, descargados de
https://primes.utm.edu/lists/small/millions/ y no-primos con por lo
menos ocho dígitos. Investiga también el efecto de la proporción de
primos y no-primos en el vector igual como, opcionalmente, la magnitud
de los números incluidos en el vector con pruebas estadísticas
adecuadas.
'''




#---------------- Var Datos_faciles  -----------------
o = 0
datos_faciles = []
#---------------- Var Datos_faciles  -----------------

from math import ceil, sqrt
def primo(n):
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(ceil(sqrt(n))), 2):
        if n % i == 0:
            return False
    return True



from scipy.stats import describe
from random import shuffle
from time import time
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import psutil

core = psutil.cpu_count()
v_core = psutil.cpu_count(logical = False)



if __name__ == "__main__":
    print()
    print("Este programa corre en una pc con", core, "nucleos, de los cuales", v_core, "son nucleos virtuales.")
    print()

    #---------------------- Importar Datos -----------------------
    with open('Datos.txt', 'r') as input:
        linea = input.readline()

    datos = [int(valor) for valor in linea.split(',')]
    num_datos = len(datos)
    print("Para este programa se han importado", num_datos, " numeros primos, desde", min(datos), "hasta", max(datos)," lo que conforma la variable datos. De esta variable se han obtenido los intervalos de numeros no-primos para generar la variable datos_faciles." )
    print("")
    #---------------------- Importar Datos -----------------------

    #---------------- De Datos genera Datos_faciles  -----------------
    for exp in range(int((num_datos + 1)/2)):
        if exp == 0:
            imp = datos[exp]
            imp2 = datos[exp + 1]
            o = exp + 1
        elif exp > 0:
            imp = datos[o + 1]
            imp2 = datos[o + 2]
            o = o + 2
        #print(imp, imp2, exp)
        for cand in range(imp, imp2+1):
            datos_faciles.append(cand)
    num_datos_faciles = len(datos_faciles)
    total_datos = num_datos + num_datos_faciles
    porcentaje_num_faciles = int((num_datos_faciles/total_datos) * 100)
    porcentaje_num_dificiles = int((num_datos/total_datos) * 100)
    print("Dado que para este programa se tienen", num_datos, "numeros primos, y", num_datos_faciles, "numeros no-primos. Por lo tanto, se tiene ", porcentaje_num_dificiles,"% de numeros primos, y un", porcentaje_num_faciles, "% de numeros no-primos." )
    print("")
    #---------------- De Datos genera Datos_faciles  -----------------

    #--------------- Genera combinaciones de Datos  ----------------
    D_F = datos
    D_F.extend(datos_faciles)
    F_D = D_F[::-1]
    Aleatorio = D_F.copy()
    shuffle(Aleatorio)
    #--------------- Genera combinaciones de Datos  ----------------
    pg1 = []
    pg2 = []
    pg3 = []
    grafica1 = []
    grafica2 = []
    grafica3 = []
    grafica4 = []
    replicas = 10
    REPLICAS = replicas - 1
    NUCLEOS = range(1, core + 1)   # Hasta 4 nucleos
    tiempos = {"ot": [], "it": [], "at": []}
    for core in NUCLEOS:
        print("-----------------------", core, "-----------------------")
        with multiprocessing.Pool(core) as pool:
            for r in range(replicas):
                t = time()
                pool.map(primo, D_F)
                tiempos["ot"].append(time() - t)
                t = time()
                pool.map(primo, F_D)
                tiempos["it"].append(time() - t)
                t = time()
                pool.map(primo, Aleatorio)
                tiempos["at"].append(time() - t)


        for tipo in tiempos:
            print("")
            print(describe(tiempos[tipo]))
            print("")
            if core == 1:
                grafica1.append(tiempos[tipo])
            elif core == 2:
                grafica2.append(tiempos[tipo])
            elif core == 3:
                grafica3.append(tiempos[tipo])
            elif core == 4:
                grafica4.append(tiempos[tipo])

        pg1.append(np.mean(tiempos["ot"]))
        pg2.append(np.mean(tiempos["it"]))
        pg3.append(np.mean(tiempos["at"]))

        tiempos = {"ot": [], "it": [], "at": []}
        print("-----------------------", core, "-----------------------")

    print("--------------------", "Global", "---------------------")
    print("")
    print(pg1)
    print("")

    print("")
    print(pg2)
    print("")

    print("")
    print(pg3)
    print("")
    print("--------------------", "Global", "---------------------")

    plt.subplot(221)
    plt.boxplot(grafica1)
    plt.xticks([1, 2, 3], ['D_F', 'F_D', 'Aleatorio'])
    plt.ylabel('Tiempo (seg)')
    plt.title('1 Núcleo')


    plt.subplot(222)
    plt.boxplot(grafica2)
    plt.xticks([1, 2, 3], ['D_F', 'F_D', 'Aleatorio'])
    plt.ylabel('Tiempo (seg)')
    plt.title('2 Núcleos')



    plt.subplot(223)
    plt.boxplot(grafica3)
    plt.xticks([1, 2, 3], ['D_F', 'F_D', 'Aleatorio'])
    plt.ylabel('Tiempo (seg)')
    plt.title('3 Núcleos')




    plt.subplot(224)
    plt.boxplot(grafica4)
    plt.xticks([1, 2, 3], ['D_F', 'F_D', 'Aleatorio'])
    plt.ylabel('Tiempo (seg)')
    plt.title('4 Núcleos')


    plt.subplots_adjust(top=0.95, bottom=0.08, left=0.05, right=0.95, hspace=0.35,
                        wspace=0.2)

    plt.show()
    plt.close()

    plt.plot(pg1, label="D_F")
    plt.xticks([0, 1, 2, 3], ['1', '2', '3', '4'])
    plt.plot(pg2, label="F_D")
    plt.xticks([0, 1, 2, 3], ['1', '2', '3', '4'])
    plt.plot(pg3, label="Aleatorio")
    plt.xticks([0, 1, 2, 3], ['1', '2', '3', '4'])
    plt.xlabel('Núcleos')
    plt.ylabel('Tiempo (seg)')
    plt.legend()
    plt.show()
    plt.close()
