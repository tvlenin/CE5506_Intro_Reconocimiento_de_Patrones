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
n_digits = 22 #For kmeans
graphics = False #Show plots with True
naive_bayes_acceptance = 0.1 # 0-1, this % from the maximum is deleted in naive bayes, 0 does nothing
normalize_param = False
svm_gamma = 100
svm_nu = 0.9

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
	for j in range(0,120):
		train_dataset.append(dataset[i+j])
for i in range(0,1351,150):
	for j in range(120,149):
		test_dataset.append(dataset[i+j])

print("\t Elapsed time: %d"%(time.time() - start))

# Apply FFT with overlapping samples
sys.stdout.write("Aplying FFT...")
sys.stdout.flush()
start = time.time()
[fft_data, audios_size, data_label] = func.fixed_size_fft(FFT_length, train_dataset, graphics)
print("\t Elapsed time: %d"%(time.time() - start))

#K means clustering
sys.stdout.write("Aplying K means...")
sys.stdout.flush()
start = time.time()
clf = KMeans(init='k-means++', n_clusters=n_digits, n_init=10).fit(fft_data)

#Plot the k-means information
func.plot_k_means(fft_data,n_digits,graphics)

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
for i in audios_size:
    b = func.bayes_predict(n_digits,clf.predict(fft_data[actual:actual+i]),naive_bayes_acceptance)
    kkk.append(b)
    actual = i
kkk = np.array(kkk)
data_label = np.array(data_label)

#Normalize the vectors
kkk = func.normalize_naive_bayes(kkk,normalize_param)
#for i in range(0,len(kkk)-1,1):
#    scaler = StandardScaler()
#    scaler.fit(kkk[i].reshape(1,-1))
#    kkk[i] = scaler.transform(kkk[i])

func.plot_some_naive_bayes(kkk,graphics,n_digits)

print("\t Elapsed time: %d"%(time.time() - start))

sys.stdout.write("Aplying Support Vector Machine...")
sys.stdout.flush()
start = time.time()
#svmm = svm.SVC(kernel='rbf', gamma = 50, C=10)
svmm = svm.NuSVC(kernel='rbf', gamma = svm_gamma ,nu = svm_nu)
svmm.fit(kkk, data_label)

##**********************************a partir de aqui predict con mi audio***************************########
kk = []
testAudio_total = []
testAudio = []
data_size1 = []
data_label1 = []
'''
for data_set in test_dataset:
    testAudio = []
    data_size1 += [(data_set[1].shape[0]-1)/128]
    data_label1 += [data_set[0]]
    for i in range(FFT_length, data_set[1].shape[0]-1, FFT_length):
    	testAudio += [np.absolute(np.fft.fft([data_set[1][(i-FFT_length):(i+FFT_length)]]))[0][0:64]]
    testAudio_total += [testAudio]
    
for i in range(290):
    b = func.bayes_predict(n_digits,clf.predict(testAudio_total[i]))
    kk.append(b)
    #print(clf.predict(testAudio_total[i]))
'''    
conta = 0
conta1 = 0
'''
for i in range (290):
    y_pred = svmm.fit(kkk, data_label).predict(kk[i].reshape(1,-1))
    if(y_pred[0] == data_label1[i]):
        conta += 1
for i in range (1200):
    y_pred = svmm.fit(kkk, data_label).predict(kkk[i].reshape(1,-1))
    if(y_pred[0] == data_label[i]):
        conta1 += 1
'''
print("\t Elapsed time: %d"%(time.time() - start))

model =svmm.fit(kkk, data_label) 
print(model.score(kkk,data_label))
#print(model.score(kk,data_label1))
#print("Error: %d "%(100-100*conta/290))
#print("Error: %d "%(100-100*conta/1200))
print("Bye")
