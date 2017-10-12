4.1 Sckit-learn
#Los datos generados por octave vienen del archivo create_data_svm.m, no se logró
#enviar parametros desde python por lo tanto si se desea cambiar el tipo de distribución
#se debe cambiar desde este archivo.
Para ejecutar el programa utilizando python 2.7 y tensorflow
$python SVM_4.1.py
#Se muestran las figuras con 3 diferentes kernel y en terminal se muestran las
#confusion_matrix
4.2 Keras y Deep Learning
#Para entrenar la red neuronal se debe ejecutar el archivo deep_learning.py de la siguiente manera
$python deep_learning.py 60000 yes
#Donde 60000 es el número de datos a entrenar y yes/no para probar la red con datos de prueba
#y mostrar la matriz de confusión.
#Una vez se haya entrenado la red su modelo se guarda en un archivo para luego cargarlo
#y probarlo cada vez que quiera ejecutar el cliente de GUI de la siguiente manera.
$python test_Mnist.py saved
#Donde se le pasa un parametro, en caso de ser "saved" prueba con un modelo entrenado con
#60000 datos y si se le pasa "new" se prueba con el modelo generado por la ultima ejecución
# de deep_learning.py
#Finalmente para entrenar la SVM con los datos MNIST se debe ejecutar
$python SVM_MNIST_4.3.py
#Esto puede tardar varios minutos.
