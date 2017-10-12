Dentro de este directorio ejecutar main() para:
1. Iniciar entrenando la red neuronal con las muestras indicadas,
    con valores aleatorios para los pesos iniciales
2. Con los pesos obtenidos se usa la funcion target para clasificar cada punto 
    de un plano, en este caso X = Y =[-1:1]. Las 3 salidas de la capa de salida
    se mapean a un color RGB para mostrar en un plano la distribucion del color.
3. La matriz de confusion cuenta en su diagonal cada vez que predice correctamente 
    para las 3 clases dadas. Fuera de la diagonal son desaciertos, la fila indica 
    la clase de entrada y la columna la clase que se predice

Para los entrenamientos realizados los pesos obtenidos generan para las 3 neuronas de salida
valores cercanos a 0.99, es decir no logra clasificar correctamente, se debe revisar la 
implemetacion del gradiente o el descenso de gradiente
