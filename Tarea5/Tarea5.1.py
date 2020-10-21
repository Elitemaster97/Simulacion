'''                             TAREA5
Determina el tamaño de muestra requerido por cada lugar decimal
de precisión del estimado obtenido para el integral, comparando
con Wolfram Alpha para por lo menos desde uno hasta siete decimales;
representa el resultado como una sola gráfica o de tipo caja-bigote
o un diagrama de violin. 
'''
from math import exp, pi
import numpy as np
def g(x):
    return (2  / (pi * (exp(x) + exp(-x))))

wolf = str(0.0488340)   #con 7 decimales

vg = np.vectorize(g)
X = np.arange(-8, 8, 0.001) # ampliar y refinar
Y = vg(X) # mayor eficiencia

from GeneralRandom import GeneralRandom
generador = GeneralRandom(np.asarray(X), np.asarray(Y))
desde = 3
hasta = 7
pedazo = 10  # desde 1 hasta 1000000 hasta
cuantos = 200 # 200

def parte(replica):
    V = generador.random(pedazo)[0]
    return ((V >= desde) & (V <= hasta)).sum()

import multiprocessing
if __name__ == "__main__":
    state = 0
    with multiprocessing.Pool(2) as pool:
        while (True):
            montecarlo = pool.map(parte, range(cuantos))

            integral = sum(montecarlo) / (cuantos * pedazo)
            num = str((pi / 2) * integral)
            pedazo = pedazo + 100


            if num[0] == wolf[0] and num[1] == wolf[1] and num[2] == wolf[2] and state == 0: # 1er Decimal
                print("Se logra el primer decimal con:", pedazo, "pedazos. Dado que:", num[2], " es igual a", wolf[2])
                state = 1

            if num[0] == wolf[0] and num[1] == wolf[1] and num[2] == wolf[2] and num[3] == wolf[3] and state == 1: # 2do Decimal
                print("Se logra el segunda decimal con:", pedazo, "pedazos. Dado que:", num[3], " es igual a", wolf[3])
                state = 2

            if num[0] == wolf[0] and num[1] == wolf[1] and num[2] == wolf[2] and num[3] == wolf[3] and num[4] == wolf[4] and state == 2: # 3er Decimal
                print("Se logra el tercer decimal con:", pedazo, "pedazos. Dado que:", num[4], " es igual a", wolf[4])
                state = 3

            if num[0] == wolf[0] and num[1] == wolf[1] and num[2] == wolf[2] and num[3] == wolf[3] and num[4] == wolf[4] and num[5] == wolf[5] and state == 3: # 4to Decimal
                print("Se logra el cuarto decimal con:", pedazo, "pedazos. Dado que:", num[5], " es igual a", wolf[5])
                state = 4

            if num[0] == wolf[0] and num[1] == wolf[1] and num[2] == wolf[2] and num[3] == wolf[3] and num[4] == wolf[4] and num[5] == wolf[5] and num[6] == wolf[6] and state == 4: # 5to Decimal
                print("Se logra el quinto decimal con:", pedazo, "pedazos. Dado que:", num[6], " es igual a", wolf[6])
                state = 5

            if num[0] == wolf[0] and num[1] == wolf[1] and num[2] == wolf[2] and num[3] == wolf[3] and num[4] == wolf[4] and num[5] == wolf[5] and num[6] == wolf[6] and num[7] == wolf[7] and state == 5: # 6to Decimal
                print("Se logra el sexto decimal con:", pedazo, "pedazos. Dado que:", num[7], " es igual a", wolf[7])
                state = 6

            if num[0] == wolf[0] and num[1] == wolf[1] and num[2] == wolf[2] and num[3] == wolf[3] and num[4] == wolf[4] and num[5] == wolf[5] and num[6] == wolf[6] and num[7] == wolf[7] and num[8] == wolf[8] and state == 6: # 7mo Decimal
                print("Se logra el septimo decimal con:", pedazo, "pedazos. Dado que:", num[8], " es igual a", wolf[8])
                break

            print(pedazo, num)
