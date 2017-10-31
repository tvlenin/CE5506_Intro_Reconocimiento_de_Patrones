import pickle
import scipy.io.wavfile as libwav
import numpy as np

#variables
#Samples for each FFT
FFT_length = 64 #for each size
overlap = 32

# 150 samples for each number, from three different people
#dataset[0-1499] -> 150 1's -> 150 2's -> ..... -> 150 9's
#dataset[n][0-1] 0-> label  1-> array normalized data
with open('dataset.dat', 'rb') as data:
    dataset = pickle.load(data)

# Apply FFT with overlapping samples
data=[]

for data_set in dataset:
    for i in range(FFT_length, data_set[1].shape[0]-1, FFT_length):
        data_fft = np.fft.fft([data_set[1][(i-FFT_length):(i+FFT_length)]])

    
