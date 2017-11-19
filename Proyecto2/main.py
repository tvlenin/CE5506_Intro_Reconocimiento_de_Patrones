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
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.svm import NuSVC

#np.random.seed(42)

#---------------variables---------------
#Samples for each FFT
FFT_length = 32 #for each size
n_digits = 16 #For kmeans
graphics = False #Show plots with True
naive_bayes_acceptance = 0 # 0-1, this % from the maximum is deleted in naive bayes, 0 does nothing
svm_gamma = 1 #no bajar, meter sesgo
svm_nu = 0.75

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
for i in range(0,1351,150):#for i in range(0,1351,150):
    for j in range(1,150):
        train_dataset.append(dataset[i+j])

for i in range(0,1351,150):
    for j in range(0,1):
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

print("%dk\t Elapsed time: %d"%(n_digits,(time.time() - start)))
# now you can save it to a file
with open('kmeans.pkl', 'wb') as f:
    pickle.dump(clf, f)

# and later you can load it
with open('kmeans.pkl', 'rb') as f:
    clf = pickle.load(f)

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

#Normalize the vectors
func.fit(kkk) #This function performs everything, for new data use func.fit_data(kk)
func.fit_data(kk)

#plt.figure(1)
#for i in range(0,999,20):
#    plt.stem(kkk[i])
#plt.show()
#plt.clf()

#plt.figure(2)
#for i in range(0,499,20):
#    plt.stem(kk[i])
#plt.show()
#plt.clf()


'''
kk1 = []
kk2 = []
for i in range(kkk.shape[0]):
    kk1 += [kkk[5:20]]
for i in range(kk.shape[0]):
    kk2 += [kk[5:20]]
'''
print("\t Elapsed time: %d"%(time.time() - start))

sys.stdout.write("Aplying Support Vector Machine...")
sys.stdout.flush()
start = time.time()

##**********************************a partir de aqui predict con mi audio***************************########
#func.fit_data(kk)

print("\t Elapsed time: %d"%(time.time() - start))
svmm = svm.NuSVC(kernel='rbf', gamma = svm_gamma ,nu = svm_nu)
model =svmm.fit(kkk, data_label)
print(model.score(kkk,data_label))
print(model.score(kk,data_label1))
#for i in range(300):
#    print(model.predict(kk[i].reshape(1,-1)))



