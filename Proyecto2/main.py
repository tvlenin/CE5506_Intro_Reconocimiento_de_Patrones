"""
==================================================================
        Reconocimiento de numeros por audio
==================================================================

Tecnologico de Costa Rica
Introduccion al Reconocimiento de Patrones  CE5506
Proyecto Final
authores: Abraham Arias, Lenin Torres

Reconocimiento de numeros en inglus del 0 al nueve por audio
En los datos utilizados tres sujetos repiten un numero diez veces
para tener 1500 datos en total
==================================================================
Si se utilizan graficos
    Se muestran 4 en total.
    Cerrarlos para continuar con la ejecucion
==================================================================
"""
print(__doc__)

import sys
import func
import time
import pickle
import numpy as np
from sklearn import svm
import scipy.io.wavfile as libwav
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.svm import NuSVC


#---------------params---------------
#Samples for each FFT. FFT_length*2 = number of samples
FFT_length = 16 # This variable indicates the overlapping, 16 means 32 samples
n_digits = 12 #For kmeans
graphics = True #Show plots with True
naive_bayes_acceptance = 0 # 0-1, this % from the maximum is deleted in naive bayes, 0 does nothing
svm_gamma = 0.01 #1/features
svm_nu = 0.5
kmeans_new = 0 # 1 to perform kmeans classification 0 to use saved

# 150 samples for each number, from three different people
#dataset[0-1499] -> 150 1's -> 150 2's -> ..... -> 150 9's
#dataset[n][0-1] 0-> label  1-> array normalized data
#open the dataset
sys.stdout.write("Opening file...")
sys.stdout.flush()
start = time.time()
with open('my_dataset2.pickle', 'rb') as data:
   dataset = pickle.load(data)

train_dataset = []
test_dataset = []
#separate the data in train and test sets
for i in range(0,1351,150):
    for j in range(50,150):
        train_dataset.append(dataset[i+j])

for i in range(0,1351,150):
    for j in range(0,50):
        test_dataset.append(dataset[i+j])

print("\t Elapsed time: %d"%(time.time() - start))

# Apply FFT with overlapping samples
sys.stdout.write("Aplying FFT...")
sys.stdout.flush()
start = time.time()
[fft_data, audios_size, data_label] = func.fixed_size_fft(FFT_length, train_dataset, graphics)#apply fft to train_dataset
[fft_data1, audios_size1, data_label1] = func.fixed_size_fft(FFT_length, test_dataset, graphics)#apply fft to test_dataset
print("\t Elapsed time: %d"%(time.time() - start))

#K means clustering
sys.stdout.write("Aplying K means...")
sys.stdout.flush()
start = time.time()
#apply Kmeans to train_dataset
if (kmeans_new == 1):
    clf = KMeans(init='random', n_clusters=n_digits, n_init=10,verbose = 0).fit(fft_data)
    # now you can save it to a file
    with open('kmeans.pkl', 'wb') as f:
        pickle.dump(clf, f)
else:
    # and later you can load it
    with open('kmeans.pkl', 'rb') as f:
        clf = pickle.load(f)
        
#func.plot_k_means(fft_data,n_digits, graphics)
print("\t Elapsed time: %d"%(time.time() - start))

sys.stdout.write("Aplying Naive Bayes...")
sys.stdout.flush()
start = time.time()
actual = 0
cont = 0
kkk = []
kk = []
#predict the phonemes with kmeans
for i in audios_size:
    b = func.bayes_predict(n_digits,clf.predict(fft_data[actual:actual+i]),naive_bayes_acceptance)
    kkk.append(b)
    actual = actual + i
actual = 0
cont = 0
for i in audios_size1:
    b = func.bayes_predict(n_digits,clf.predict(fft_data1[actual:actual+i]),naive_bayes_acceptance)
    kk.append(b)
    actual = actual + i
kkk = np.array(kkk)
kk = np.array(kk)

#get the label of data
data_label = np.array(data_label)
data_label1 = np.array(data_label1)


#Normalize the vectors
#Not performing this
#func.fit(kkk)
#func.fit_data(kk)

func.plot_some_naive_bayes(kkk,graphics,n_digits)
print("\t Elapsed time: %d"%(time.time() - start))
sys.stdout.write("Aplying Support Vector Machine...")
sys.stdout.flush()
start = time.time()


print("\t Elapsed time: %d"%(time.time() - start))
svmm = svm.NuSVC(kernel='rbf', gamma = svm_gamma, nu=svm_nu )
model =svmm.fit(kkk, data_label)# train the SVM
print("Train score:")
print(model.score(kkk,data_label))
print("Test score:")
print(model.score(kk,data_label1))

y = model.predict(kk)
print(confusion_matrix(data_label1, y))

print("Bye")
