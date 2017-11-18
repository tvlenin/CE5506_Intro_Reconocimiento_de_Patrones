'''
Normalizacon por la suma distribucion(todo debe quedar entre 0 - 1)
Volver a ver gamma y nu


'''

import sys
import func
import time
import pickle
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import scipy.io.wavfile as libwav
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVC

np.random.seed(42)

#---------------variables---------------
#Samples for each FFT
FFT_length = 64 #for each size
n_digits = 26 #For kmeans
graphics = False #Show plots with True
naive_bayes_acceptance = 0 # 0-1, this % from the maximum is deleted in naive bayes, 0 does nothing
normalize_param = False
svm_gamma = 1 #no bajar, meter sesgo
svm_nu = 0.1
svm_c = 40

# 150 samples for each number, from three different people
#dataset[0-1499] -> 150 1's -> 150 2's -> ..... -> 150 9's
#dataset[n][0-1] 0-> label  1-> array normalized data
sys.stdout.write("Opening file...")
sys.stdout.flush()
start = time.time()
with open('dataset.dat', 'rb') as data:
   dataset = pickle.load(data)

train_dataset = []
test_dataset = []
for i in range(0,1351,150):
    for j in range(50,150):
        train_dataset.append(dataset[i+j])

for i in range(0,1351,150):
    for j in range(49,100):
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
clf = KMeans(init='k-means++', n_clusters=n_digits, n_init=10).fit(fft_data)

print("\t Elapsed time: %d"%(time.time() - start))
# now you can save it to a file
#with open('kmeans.pkl', 'wb') as f:
#    pickle.dump(clf, f)

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
    b = func.bayes_predict(n_digits,clf.predict(fft_data[actual:actual+i]),naive_bayes_acceptance)
    kkk.append(b)
    actual = i
actual = 0
for i in audios_size1:
    b = func.bayes_predict(n_digits,clf.predict(fft_data1[actual:actual+i]),naive_bayes_acceptance)
    kk.append(b)
    actual = i
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



#kk = func.normalize_naive_bayes(kk,normalize_param)

print("\t Elapsed time: %d"%(time.time() - start))
svmm = svm.NuSVC(kernel='rbf', gamma = svm_gamma ,nu = svm_nu)
model =svmm.fit(kkk, data_label)
print(model.score(kkk,data_label))
print(model.score(kk,data_label1))


'''
for i in range(100):
    for j in range(200):
        svmm = svm.NuSVC(kernel='rbf', gamma = (svm_gamma+i*0.001) ,nu = (svm_nu+j*0.0001))
        model =svmm.fit(kkk, data_label)
        if(model.score(kk,data_label1) > 0.17):
            print( svm_gamma+i*0.001)
            print( svm_nu+j*0.00001)
            print(model.score(kkk,data_label))
            print(model.score(kk,data_label1))
'''
print("Bye")
