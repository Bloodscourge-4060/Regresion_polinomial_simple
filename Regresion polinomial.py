import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Datos.csv')  
x = df['araba_fiyat'] 
y = df['araba_max_hiz'] 

# Grado del polinomio
n =4 

# Realiza la regresión polinomial
poly_coeff = np.polyfit(x, y, n)

# Crea una función de polinomio con los coeficientes
poly_func = np.poly1d(poly_coeff)

# Genera puntos para graficar la regresión
x_pred = np.linspace(x.min(), x.max(), 100)
y_pred = poly_func(x_pred)

# Grafica los datos y la regresión polinomial
plt.scatter(x, y, label='Datos')
plt.plot(x_pred, y_pred, 'r', label=f'Regresión Polinomial (Grado {n})')
plt.xlabel('araba_fiyat')
plt.ylabel('araba_max_hiz')
plt.legend()
plt.show()
