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
print(dataset[0])
train_dataset = []
test_dataset = []
for i in range(0,1351,150):
	for j in range(0,120):
		train_dataset.append(dataset[i+j])
for i in range(0,1351,150):
	for j in range(120,149):
		test_dataset.append(dataset[i+j])

print(test_dataset[0])
# Apply FFT with overlapping samples
#print("Aplying FFT...")
#[fft_data, audios_size, data_label] = fft.fixed_size_fft(FFT_length, dataset)
