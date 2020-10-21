'''                             TAREA5
Aplica un método Monte Carlo para estimar la cantidad de pintura
necesaria en un mural, comparando conteos exactos de pixeles de
distintos colores (recomiendo discretizar a un palette de pocos
colores) con conteos estimados con muestreo aleatorio. Completando
ambas opciones permite puntajes hasta ocho en la tarea base.
'''
from collections import Counter
from random import randint
from PIL import Image
import matplotlib.pyplot as plt
import sys

try:
    Imagen = Image.open("colores.jpg")

except IOError:
    print("Unable to load image")
    sys.exit(1)


size = w, h = Imagen.size
data = Imagen.load()

total_pixel = w * h

print("\nLa imagen a porcesar tiene las siguientes caracteristicas:\n")
print(" Format: {0}\n Size: {1}\n Mode: {2}".format(Imagen.format,
    Imagen.size, Imagen.mode), "\n")

print("Por lo tanto, para esta imagen se tiene un total de", total_pixel, "pixeles.\n")

Global_R, Global_G, Global_B = [], [], []

colors = []
for x in range(w):
    for y in range(h):

        colores = data[x,y]
        Ra, Ga, Ba = colores[0], colores[1], colores[2]

        sum_RGB = (Ra + Ga + Ba)
        porcentaje_R = ((Ra * 100)/sum_RGB)
        porcentaje_G = ((Ga * 100)/sum_RGB)
        porcentaje_B = ((Ba * 100)/sum_RGB)
        Lt_R = ((porcentaje_R)/ 100)
        Lt_G = ((porcentaje_G)/ 100)
        Lt_B = ((porcentaje_B)/ 100)
        Global_R.append(Lt_R)
        Global_G.append(Lt_G)
        Global_B.append(Lt_B)



R = sum(Global_R)
G = sum(Global_G)
B = sum(Global_B)

print("Esta imagen necesita la siguiente cantidad de pintura, suponiendo que para pintar 1 pixel necesitas 1 litro de pintura:\n R:{0}L. \n G:{1}L. \n B:{2}L.".format(R, G, B))


repeticiones = 0
Global_r, Global_g, Global_b = [], [], []
grafica_R, grafica_G, grafica_B = [], [], []
PORCENTAJES_R, PORCENTAJES_G, PORCENTAJES_B = 0, 0, 0


while (True):
    repeticiones = repeticiones + 1

    r = randint(0, 256)
    g = randint(0, 256)
    b = randint(0, 256)


    sum_rgb = (r + g + b)
    porcentaje_r = ((r * 100)/sum_rgb)
    porcentaje_g = ((g * 100)/sum_rgb)
    porcentaje_b = ((b * 100)/sum_rgb)
    if PORCENTAJES_R <= 100:
        Lt_r = ((porcentaje_r)/ 100)
    else:
        Lt_r = 0
    if PORCENTAJES_G <= 100:
        Lt_g = ((porcentaje_g)/ 100)
    else:
        Lt_g = 0
    if PORCENTAJES_B <= 100:
        Lt_b = ((porcentaje_b)/ 100)
    else:
        Lt_b = 0
    Global_r.append(Lt_r)
    Global_g.append(Lt_g)
    Global_b.append(Lt_b)
    Total_r = sum(Global_r)
    Total_g = sum(Global_g)
    Total_b = sum(Global_b)
    PORCENTAJES_R = ((Total_r * 100) / R )
    PORCENTAJES_G = ((Total_g * 100) / G )
    PORCENTAJES_B = ((Total_b * 100) / B )

    grafica_R.append(PORCENTAJES_R)

    grafica_G.append(PORCENTAJES_G)

    grafica_B.append(PORCENTAJES_B)


    if PORCENTAJES_R >= 100 and PORCENTAJES_G >= 100 and PORCENTAJES_B >= 100:
        break
#print( PORCENTAJES_R, PORCENTAJES_G, PORCENTAJES_B )

limite = [100] * repeticiones

plt.plot(grafica_R, 'r', label="Rojo")
#plt.xticks([0, 1, 2, 3], ['1', '2', '3', '4'])
plt.plot(grafica_G, 'g', label="Verde")
#plt.xticks([0, 1, 2, 3], ['1', '2', '3', '4'])
plt.plot(grafica_B, 'b', label="Azul")
#plt.xticks([0, 1, 2, 3], ['1', '2', '3', '4'])
plt.plot(limite, 'k--', )
plt.xlabel('Iteraciones ')
plt.ylabel('Porcentaje')
plt.legend()
plt.show()
plt.close()
