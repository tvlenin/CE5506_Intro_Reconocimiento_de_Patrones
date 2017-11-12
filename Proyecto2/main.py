import pickle
import scipy.io.wavfile as libwav
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import svm

import fft
import func
np.random.seed(42)

#---------------variables---------------
#Samples for each FFT
FFT_length = 64 #for each size
FFT_size = 64
n_digits = 30 #For kmeans

# 150 samples for each number, from three different people
#dataset[0-1499] -> 150 1's -> 150 2's -> ..... -> 150 9's
#dataset[n][0-1] 0-> label  1-> array normalized data
print("Opening file...")
with open('dataset.dat', 'rb') as data:
   dataset = pickle.load(data)

train_dataset = []
test_dataset = []
for i in range(0,1351,150):
	for j in range(0,120):
		train_dataset.append(dataset[i+j])
for i in range(0,1351,150):
	for j in range(120,149):
		test_dataset.append(dataset[i+j])

# Apply FFT with overlapping samples
print("Aplying FFT...")
[fft_data, audios_size, data_label] = fft.fixed_size_fft(FFT_length, train_dataset)

#K means clustering
print("Aplying K means")
#clf = KMeans(init='k-means++', n_clusters=n_digits, n_init=10).fit(fft_data)


# now you can save it to a file
#with open('kmeans.pkl', 'wb') as f:
#    pickle.dump(clf, f)

# and later you can load it
with open('kmeans.pkl', 'rb') as f:
    clf = pickle.load(f)
actual = 0
cont = 0
kkk = []
for i in audios_size:
    #print(clf.predict(fft_data[actual:actual+i]))
    b = np.zeros(160)
    b[0:i] = clf.predict(fft_data[actual:actual+i])
    #print(b)#aqui veo lo que tira el predict de kmeans de los datos de entrenamiento
    kkk.append(b)
    actual = i
    #cont += 1
kkk = np.array(kkk)
data_label = np.array(data_label)
print(kkk.shape)
print(data_label.shape)

svmm = svm.SVC(kernel='rbf', gamma = 10)
svmm.fit(kkk, data_label)
##**********************************a partir de aqui predict con mi audio***************************########
#a1 = libwav.read('/home/tvlenin/Desktop/1_Lenin_0.wav',mmap=False)[1]
#print(a1.shape)
#a1 = preprocessing.normalize(a1.reshape(-1,1), norm='l2')
#print(a1.shape)
kk = []
testAudio = []
data_size1 = []
data_label1 = []
for data_set in test_dataset:
	data_size1 += [(data_set[1].shape[0]-1)/128]
	data_label1 += [data_set[0]]
	for i in range(FFT_length, data_set[1].shape[0]-1, FFT_length):
		testAudio += [np.absolute(np.fft.fft([data_set[1][(i-FFT_length):(i+FFT_length)]]))[0][0:64]]


for i in range(300):
    #print(clf.predict(fft_data[actual:actual+i]))
    b = np.zeros(160)
    testAudio_test = np.array(testAudio[i])
    #print(testAudio_test)
    b[0:testAudio_test.shape[0]] = clf.predict(testAudio_test.reshape(1,testAudio_test.shape[0]))
    kk.append(b)

#testAudio = np.array(testAudio)
#print(testAudio.shape[0])
#b[0:testAudio.shape[0]] = clf.predict(testAudio.reshape(testAudio.shape[0],testAudio.shape[1]))
##*************************************************************************************************########
#print(b)
#print(kkk[73])
#kk.append(b)

#y_pred = svmm.fit(kkk, data_label).predict(kkk[73].reshape(1,-1))
for i in range (300):
	#y_pred = svmm.fit(kkk, data_label).predict(kk[i].reshape(1,-1))
	print(kk[i])#aqui veo lo que tira el predict de kmeans de los datos de prueba


print("Bye")
