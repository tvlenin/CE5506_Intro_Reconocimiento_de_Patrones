import pickle
import scipy.io.wavfile as libwav
import numpy as np

#variables
#Samples for each FFT
#FFT_length = 5

# 150 samples for each number, from three different people
#dataset[0-1499] -> 150 1's -> 150 2's -> ..... -> 150 9's
#dataset[n][0-1] 0-> label  1-> array normalized data
with open('dataset.dat', 'rb') as data:
    dataset = pickle.load(data)

# Apply FFT with overlapping samples
data=[]
for data_set in dataset:
    data_fft = np.fft.fft(data_set[1])
    data+=[data_set[0], data_fft]


    
