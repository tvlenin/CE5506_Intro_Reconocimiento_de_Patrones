'''
Normalizacon por la suma distribucion(todo debe quedar entre 0 - 1)
Volver a ver gamma y nu
comparar las normalizaciones entre los dos datos
'''

import sys
import func
import time
import pickle
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import scipy.io.wavfile as libwav
from sklearn import neighbors,metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import NuSVC

#np.random.seed(42)

#---------------variables---------------
#Samples for each FFT
FFT_length = 16 #for each size
n_digits = 12 #For kmeans
graphics = False #Show plots with True
naive_bayes_acceptance = 0.1 # 0-1, this % from the maximum is deleted in naive bayes, 0 does nothing
svm_gamma = 0.01 #no bajar, meter sesgo
svm_nu = 0.5
svm_c = 40

# 150 samples for each number, from three different people
#dataset[0-1499] -> 150 1's -> 150 2's -> ..... -> 150 9's
#dataset[n][0-1] 0-> label  1-> array normalized data
sys.stdout.write("Opening file...")
sys.stdout.flush()
start = time.time()
with open('my_dataset2.pickle', 'rb') as data:
   dataset = pickle.load(data)

train_dataset = []
test_dataset = []
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
[fft_data, audios_size, data_label] = func.fixed_size_fft(FFT_length, train_dataset, graphics)
[fft_data1, audios_size1, data_label1] = func.fixed_size_fft(FFT_length, test_dataset, graphics)
print("\t Elapsed time: %d"%(time.time() - start))

#K means clustering
sys.stdout.write("Aplying K means...")
sys.stdout.flush()
start = time.time()
#print(fft_data)
clf = KMeans(init='random', n_clusters=n_digits, n_init=10,verbose = 0).fit(fft_data)

print("\t Elapsed time: %d"%(time.time() - start))
# now you can save it to a file
with open('kmeans.pkl', 'wb') as f:
    pickle.dump(clf, f)

# and later you can load it
#with open('kmeans.pkl', 'rb') as f:
#    clf = pickle.load(f)

sys.stdout.write("Aplying Naive Bayes...")
sys.stdout.flush()
start = time.time()
actual = 0
cont = 0
kkk = []
kk = []
for i in audios_size:
    b = func.bayes_predict(n_digits,clf.predict(fft_data[actual:actual+i]),naive_bayes_acceptance)#func.bayes_predict(n_digits,clf.predict(fft_data[actual:actual+i]),naive_bayes_acceptance)
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
data_label = np.array(data_label)
data_label1 = np.array(data_label1)

func.plot_some_naive_bayes(kkk,graphics,n_digits)
print("\t Elapsed time: %d"%(time.time() - start))

sys.stdout.write("Aplying Support Vector Machine...")
sys.stdout.flush()
start = time.time()

##**********************************a partir de aqui predict con mi audio***************************########
func.fit(kkk)
func.fit_data(kk)

print("\t Elapsed time: %d"%(time.time() - start))
svmm = svm.NuSVC(kernel='rbf', gamma = svm_gamma, nu=svm_nu )
model =svmm.fit(kkk, data_label)
print(model.score(kkk,data_label))
print(model.score(kk,data_label1))
#for i in range(300):
#    print(model.predict(kk[i].reshape(1,-1)))
print("Bye")
