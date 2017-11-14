import pickle
import fft
import func
import numpy as np
import scipy.io.wavfile as libwav
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB #Naive bayes

np.random.seed(42)

#------------------------------------Algorithm parameters-----------------------------------------------
FFT_length = 256 #Samples for each FFT
n_digits = 26 #For kmeans

''' 150 samples for each number, from three different people
dataset[0-1499] -> 150 1's -> 150 2's -> ..... -> 150 9's
dataset[n][0-1] 0-> label  1-> array normalized data'''
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

#-----------------------------------FFT with overlapping samples-----------------------------------------------
print("Aplying FFT...")
[fft_data, audios_size, data_label] = fft.fixed_size_fft(FFT_length, train_dataset)

#------------------------------------K means clustering--------------------------------------------------------
print("Aplying K means")
#clf = KMeans(init='k-means++', n_clusters=n_digits, n_init=10).fit(fft_data)

#-----------------------------------File write/read k-means----------------------------------------------------
# now you can save it to a file
#with open('kmeans.pkl', 'wb') as f:
#    pickle.dump(clf, f)

# and later you can load it
with open('kmeans.pkl', 'rb') as f:
    clf = pickle.load(f)


#-----------------------------------Naive Bayes----------------------------------------------------
actual = 0
kkk = []
for i in audios_size:
    #print(clf.predict(fft_data[actual:actual+i]))
    b = np.zeros(160)
    b[0:i] = clf.predict(fft_data[actual:actual+i])
    kkk.append(b)
    actual = i
kkk = np.array(kkk)
data_label = np.array(data_label)

gnb = GaussianNB()
naive_bayes = gnb.fit(kkk,data_label)


kk = []
testAudio_total = []
testAudio = []
data_size1 = []
data_label1 = []
for data_set in test_dataset:
    testAudio = []
    data_size1 += [(data_set[1].shape[0]-1)/128]
    data_label1 += [data_set[0]]
    for i in range(FFT_length, data_set[1].shape[0]-1, FFT_length):
    	testAudio += [np.absolute(np.fft.fft([data_set[1][(i-FFT_length):(i+FFT_length)]]))[0][0:64]]
    testAudio_total += [testAudio]

for i in range(0,290,10):
    b = np.zeros(160)
    b[0:len(testAudio_total[i])] = clf.predict(testAudio_total[i])
    kk.append(b)


for i in range (29):
	y_pred = naive_bayes.predict(kk[i].reshape(1,-1))
	print(y_pred)
    #print(kk[i])
















