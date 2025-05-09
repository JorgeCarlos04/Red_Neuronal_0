import numpy as np
import matplotlib.pyplot as plt

class Red_Neuronal_Lineal:
    def __init__(self, entradas, tasa_aprendizaje=0.01):
        self.tasa_aprendizaje = tasa_aprendizaje
        self.pesos = np.random.rand(entradas, 1) * 0.1  # Corregido 'entrada' a 'entradas'
        self.sesgos = np.zeros((1, 1))
   
    def act_lineal(self, x):
        return x
   
    def dev_lineal(self, x):
        return np.ones_like(x)  # Corregido 'one' a 'ones_like'

    def forward(self, X):
        self.red = np.dot(X, self.pesos) + self.sesgos 
        self.salida = self.act_lineal(self.red)
        return self.salida

    def backward(self, entrada, salida_esp):
        error = self.salida - salida_esp
        delta = error * self.dev_lineal(self.red)  # Corregido 'self.z' a 'self.red'
        grad_pesos_med = np.dot(entrada.T, delta) / len(entrada)  # Corregido 'X' a 'entrada'
        grad_sesgo_med = np.sum(delta, axis=0) / len(entrada)  # Corregido 'axon' a 'axis'

        self.pesos -= self.tasa_aprendizaje * grad_pesos_med
        self.sesgos -= self.tasa_aprendizaje * grad_sesgo_med
        return np.mean(error ** 2)

    def Entrenamiento(self, X, Y, tiempo):  # Corregido 'x, y' a 'X, Y'
        historial_tiempo = []
        for i in range(tiempo):
            self.forward(X)
            error = self.backward(X, Y)
            historial_tiempo.append(error)
        return historial_tiempo
    
    def predecir(self, X):
        return self.forward(X)

    def grafica_plt(self, historial_tiempo):
        plt.figure(figsize=(8,5))
        plt.plot(historial_tiempo, label="Error**2")
        plt.xlabel("Tiempo")  # Corregido 'Xlabel' a 'xlabel'
        plt.ylabel("Error")  # Corregido 'plo.Ylabel' a 'plt.ylabel'
        plt.title("Evolucion del error a traves del tiempo")
        plt.legend()  # Añadidos paréntesis
        plt.show()



# Datos de prueba (fuera de la clase)
ent = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])  # Corregida la sintaxis
sal_espe = np.array([[0.1], [0.5], [0.8]])  # Corregido '0,8' a '0.8'

red = Red_Neuronal_Lineal(entradas=3, tasa_aprendizaje=0.01)  # Corregido 'entrada' a 'entradas'
historial = red.Entrenamiento(ent, sal_espe, tiempo=1000)
red.grafica_plt(historial)

muestra = np.array([[0.1], [0.5], [0.8]])  # Corregida la sintaxis
prediccion = red.predecir(muestra)
print(f"Prediccion para {muestra[0]}: {prediccion[0][0]}")  # Corregidas las comillas