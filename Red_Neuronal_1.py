import numpy as np
import matplotlib.pyplot as plt

class Red_Neuronal_Lineal:
    def __init__(self, entradas, tasa_aprendizaje=0.01):
        self.tasa_aprendizaje = tasa_aprendizaje
        self.pesos = np.random.rand(entradas, 1) * 0.1  
        self.sesgos = np.zeros((1, 1))
   
    def act_lineal(self, x):
        return x
   
    def dev_lineal(self, x):
        return np.ones_like(x)  

    def forward(self, X):
        self.red = np.dot(X, self.pesos) + self.sesgos 
        self.salida = self.act_lineal(self.red)
        return self.salida

    def backward(self, entrada, salida_esp):
        error = self.salida - salida_esp
        delta = error * self.dev_lineal(self.red)  
        grad_pesos_med = np.dot(entrada.T, delta) / len(entrada)  
        grad_sesgo_med = np.sum(delta, axis=0) / len(entrada)  

        self.pesos -= self.tasa_aprendizaje * grad_pesos_med
        self.sesgos -= self.tasa_aprendizaje * grad_sesgo_med
        return np.mean(error ** 2)

    def Entrenamiento(self, X, Y, tiempo):  
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
        plt.plot(historial_tiempo, 
        plt.xlabel("Tiempo")  
        plt.ylabel("Error")
        plt.title("Evolucion del error a traves del tiempo")
        plt.legend()  
        plt.show()



