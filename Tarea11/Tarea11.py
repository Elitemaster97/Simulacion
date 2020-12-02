'''                             TAREA11
Grafica el porcentaje de soluciones de Pareto como función del número
de funciones objetivo para k ∈ [2, 12] con diagramas de violín
combinados con diagramas de caja-bigote, verificando que diferencias
observadas, cuando las haya, sean estadísticamente significativas.
Razona en escrito a qué se debe el comportamiento observado.
'''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from random import randint, random


def poli(maxdeg, varcount, termcount):
    f = []
    for t in range(termcount):
        var = randint(0, varcount - 1)
        deg = randint(1, maxdeg)
        f.append({'var': var, 'coef': random(), 'deg': deg})
    return pd.DataFrame(f)

def evaluate(pol, var):
    return sum([t.coef * var[pol.at[i, 'var']]**t.deg for i, t in pol.iterrows()])


def domin_by(target, challenger):
    if np.any(np.greater(target, challenger)):
        return False
    return np.any(np.greater(challenger, target))

def porcentaje(f,total):
    return ((f * 100) / total)

def simu(k):
    vc = 4
    md = 3
    tc = 5
    obj = [poli(md, vc, tc) for i in range(k)]
    minim = np.random.rand(k) > 0.5
    n = 100 # cuantas soluciones aleatorias
    sol = np.random.rand(n, vc)
    val = np.zeros((n, k))
    for i in range(n): # evaluamos las soluciones
        for j in range(k):
            val[i, j] = evaluate(obj[j], sol[i])

    sign = [1 + -2 * m for m in minim]

    no_dom = []
    for i in range(n):
        d = [domin_by(sign * val[i], sign * val[j]) for j in range(n)]
        no_dom.append(not np.any(d)) # si es cierto que ninguno es verdadero
    frente = val[no_dom, :]

    porcent = porcentaje(len(frente),n)
    return porcent


rep = 40
POR = []
for k in range(2,13):
    for i in range(rep):
        print("----",k,",",i,"----")
        s = simu(k)
        POR.append(s)


df = pd.DataFrame(
    {"Objetivos": rep * ["2"] + rep * ["3"] + rep * ["4"] + rep * ["5"] + rep * ["6"] + rep * ["7"] + rep * ["8"] + rep * ["9"] + rep * ["10"] + rep * ["11"] + rep * ["12"],
     "Porcentajes": POR}
     )

pd.set_option("display.max_rows", None, "display.max_columns", None)
print(df)
print(POR)

sns.violinplot(x='Objetivos', y='Porcentajes', data=df, scale='count', cut = 0)
sns.swarmplot(x="Objetivos", y="Porcentajes", data=df, color=".25")
plt.show()
