import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# Datos de ejemplo: temperatura y precio
temperatura = np.array([25, 30, 35, 20, 15, 27, 32, 18, 23, 28])
precio = np.array([3, 4, 5, 2, 1, 3.5, 4.2, 1.8, 2.5, 3.2])
ventas = np.array([200, 230, 260, 180, 150, 210, 240, 170, 190, 220])

# Crear una instancia del modelo SVR
svm = SVR(kernel='linear')


# Entrenar el modelo con los datos de entrada
svm.fit(np.column_stack((temperatura, precio)), ventas)

# Datos de entrada para predecir las ventas
nuevas_temperaturas = np.array([28, 33, 25, 22])
nuevos_precios = np.array([3.2, 4.5, 2.8, 2.1])

# Realizar las predicciones de ventas
predicciones_ventas = svm.predict(np.column_stack((nuevas_temperaturas, nuevos_precios)))

# Crear el gráfico de dispersión
plt.scatter(temperatura, precio, c=ventas, cmap='viridis', label='Datos reales')
plt.scatter(nuevas_temperaturas, nuevos_precios, c=predicciones_ventas, cmap='viridis', marker='s', label='Predicciones')
plt.colorbar(label='Ventas')
plt.xlabel('Temperatura')
plt.ylabel('Precio')
plt.title('Predicciones de ventas de helados')
plt.legend()
plt.show()
