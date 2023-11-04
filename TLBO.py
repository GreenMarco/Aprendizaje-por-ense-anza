##%
#Importaciones
import matplotlib.pyplot as plt
import numpy as np
from Plot_Surf import plot_surf
from Plot_Contour import plot_contour

#%%
#Funciones
def Penalty(x, xl, xu):
    D = x.size
    z = 0

    for j in range(D):
        if x[j] < xl[j]:
            z = z + 1
        elif x[j] > xu[j]:
            z = z + 1
        else:
            z = z + 0

#%%
#Funcion objetivo

# McCormick Function
f = lambda x, y: np.sin(x+y) + (x-y) ** 2 - 1.5 * x + 2.5 * y+1 

#Sphere
#f = lambda x, y: (x)**2 + (y)**2

#Rastring
#f = lambda x, y: 10*2 + x**2 + y**2 - 10*np.cos(2*np.pi*x) - 10*np.cos(2*np.pi*y)

#Griewank
#f = lambda x, y: ((x**2/4000)+(y**2/4000))-(np.cos(x)*np.cos(y/np.sqrt(2)))+1

fp = lambda x, xl, xu: f(x[0], x[1]) + 1000 * Penalty(x, xl, xu);
#%%
#Parametros
xl = np.array([-5, -5])
xu = np.array([5, 5])

D = 2
N = 30
G = 100

x = np.zeros((D, N))
fitness = np.zeros(N)

#%%
#Algoritmo
for i in range(N):
    x[:, i] = xl + (xu - xl) * np.random.rand(D)
    fitness[i] = fp(x, xl,xu)

for g in range(G):
    plot_contour(f, x, xl, xu)

    for i in range(N):
        # Teacher phase
        t = np.argmin(fitness)
        Tf = np.random.randint(2)
        c = np.zeros(D)

        for j in range(D):
            x_mean = np.mean(x[j, :])
            r = np.random.rand()

            c[j] = x[j, i] + r * (x[j, t] - Tf * x_mean)

        fc = f(c[0], c[1])

        if fc < fitness[i]:
            x[:, i] = c
            fitness[i] = fc

        # Learner phase
        k = i
        while k == i:
            k = np.random.randint(N)

        c = np.zeros(D)

        if fitness[i] < fitness[k]:
            for j in range(D):
                r = np.random.rand()
                c[j] = x[j, i] + r * (x[j, i] - x[j, k])
        else:
            for j in range(D):
                r = np.random.rand()
                c[j] = x[j, i] + r * (x[j, k] - x[j, i])

        fc = f(c[0], c[1])

        if fc < fitness[i]:
            x[:, i] = c
            fitness[i] = fc

igb = np.argmin(fitness)

print("Mínimo global en x=", x[0, igb], " y=", x[1, igb], " f(x,y)=", f(x[0, igb], x[1, igb]))
plot_surf(f, x, xl, xu, igb)
