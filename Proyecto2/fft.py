import numpy as np

'''
Apply FFT with overlapping samples
Apply FFT all the samples with overlap
IF pass 64 for length, does a overlap of 32 and takes 128 samples
Returns [label, fft_overlaped] for each sound
'''
def fixed_size_fft(FFT_length, dataset):
    data = []
    data_fft = []
    data_size = []
    data_label = []
    for data_set in dataset:
        data_size += [(data_set[1].shape[0]-1)/128]
        data_label += [data_set[0]]
        for i in range(FFT_length, data_set[1].shape[0]-1, FFT_length):
            data_fft += [np.absolute(np.fft.fft([data_set[1][(i-FFT_length):(i+FFT_length)]]))[0][0:64]]
#            if(count==0):
#                print(np.absolute(np.fft.fft([data_set[1][(i-FFT_length):(i+FFT_length)]]))[0][0:63].shape)
#            count+=1
        #data+=[[data_set[0], data_fft]]
        #data_fft=[]
    return data_fft, data_size, data_label


def standart_size_fft(number_FFTs, dataset):
    data=[]
    data_fft=[]
    #Work for each audio, which contains [label, audio]
    for data_item in dataset:
        samples_per_FFT = data_item[1].shape[0]/number_FFTs
        for i in range(0, data_item[1].shape[0]-1, samples_per_FFT):
            data_fft += [np.absolute(np.fft.fft(data_item[1][(i):(i+samples_per_FFT-1)]))]
        data+=[data_fft] # Without [data_item[0] can't concatenate it for kmeans
        data_fft=[]

    return data
