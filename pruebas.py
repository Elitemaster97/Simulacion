from random import random, randint

x = 4
y = 4

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

def experimento(largo, dim, x, y):
    pos = [0] * dim     #Generando vector de posicion en el origen
    for t in range(largo):        # Cantidad de pasos a realizar
        x = contadorx(x,y)       # Contador de 5 a 10
        y = contadory(y)        # Memoria del contador x
        pos = paso(pos, dim)     # Actualizamos pos
        if all([p == 0 for p in pos]):
            return True
    return False


rep = range(1,9)     # Dimensiones a probar
largo = 100
total = 1000
regresos = 0
print("")
print("%Regresos Dimension")
for i in rep:
    regresos = 0
    dim = i
    for replica in range(total):
        regresos += experimento(largo, dim, x, y)
    print(regresos / total, i)
print("")
