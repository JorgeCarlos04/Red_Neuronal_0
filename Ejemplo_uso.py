from Red_Neuronal import Red_Neuronal_Lineal


ent = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])  
sal_espe = np.array([[0.1], [0.5], [0.8]])  
red = Red_Neuronal_Lineal(entradas=3, tasa_aprendizaje=0.01)  
historial = red.Entrenamiento(ent, sal_espe, tiempo=1000)
red.grafica_plt(historial)
muestra = np.array([[0.1], [0.5], [0.8]])  
prediccion = red.predecir(muestra)
print(f"Prediccion para {muestra[0]}: {prediccion[0][0]}")  