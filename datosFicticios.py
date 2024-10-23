import numpy as np
from sklearn.linear_model import LinearRegression
import sympy as sp

# Datos simulados de tasas de desempleo (por ejemplo, años 2010-2024)
anios = np.array([2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]).reshape(-1, 1)
tasas_desempleo = np.array([4.1, 3.9, 4.0, 3.8, 3.6, 3.7, 3.9, 4.2, 4.5, 4.7, 6.2, 5.4, 4.8, 3.8, 3.0])

# Crear el modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(anios, tasas_desempleo)

# Predicción inicial para el 2025
anio_pred = np.array([[2025]])
prediccion_inicial = modelo.predict(anio_pred)
print(f"Predicción inicial para 2025: {prediccion_inicial[0]:.2f}%")

# Aplicando el método de Newton para optimizar los parámetros
# Definimos las variables
x = sp.Symbol('x')
y_real = prediccion_inicial[0]  # Predicción inicial
y_pred = modelo.coef_[0] * x + modelo.intercept_  # Ecuación de la regresión

# Definimos la función de error (error cuadrático)
error = (y_pred - y_real)**2

# Derivadas
derivada = sp.diff(error, x)

# Método de Newton para encontrar el valor óptimo
def newton_raphson(derivada, valor_inicial, tol=1e-6, max_iter=100):
    x_n = valor_inicial
    for _ in range(max_iter):
        f_val = derivada.subs(x, x_n)
        f_prime = sp.diff(derivada, x).subs(x, x_n)
        if abs(f_val) < tol:
            break
        x_n = x_n - f_val / f_prime
    return x_n

# Ajustamos los parámetros utilizando el método de Newton
ajuste_parametros = newton_raphson(derivada, valor_inicial=2025)
print(f"Predicción ajustada para 2025 tras optimización: {ajuste_parametros:.2f}%")
