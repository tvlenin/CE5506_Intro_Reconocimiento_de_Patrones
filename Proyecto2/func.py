import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import math


def extend_zeros(data):
    #find the biggest audio
    max_size=0
    for item in data:
        if(np.shape(item[1])[0]>max_size):
            max_size = np.shape(item[1])[0]
    #extend the others
    for item in data:
        item[1] = item[1] + np.zeros(shape=(max_size-np.shape(item[1])[0], 1))

    return data

'''
Apply FFT with overlapping samples
Apply FFT all the samples with overlap
IF pass 64 for length, does a overlap of 32 and takes 128 samples
Returns [label, fft_overlaped] for each sound
'''
def fixed_size_fft(FFT_length, dataset, plot):
    cont = 0
    data = []
    data_fft = []
    data_size = []
    data_label = []
    for data_set in dataset:
        data_size += [(data_set[1].shape[0])/FFT_length]

        data_label += [data_set[0]]
        for i in range(FFT_length, data_set[1].shape[0]-1, FFT_length):
            a= np.absolute(np.fft.fft([data_set[1][(i-FFT_length):(i+FFT_length)]]))[0]
            cont += 1
            if(cont == 0 and plot==True):
                cont +=1
                plt.figure(1)
                plt.plot(np.absolute(([data_set[1][(i-FFT_length):(i+FFT_length)]]))[0])
                plt.ylabel('Amplitude')
                plt.xlabel('t')
                plt.title('128 Samples of the audio')
                plt.show()
                plt.figure(2)
                plt.plot(a)
                plt.ylabel('Amplitude')
                plt.xlabel('f')
                plt.title('FFT')
                plt.show()

            data_fft += [a[0:FFT_length]]



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


def plot_k_means(fft_data,n_digits, graphics):
    if(graphics==True):
        h = .01
        reduced_data = PCA(2).fit_transform(fft_data)
        clf_reduced = KMeans(init='k-means++', n_clusters=n_digits, n_init=10).fit(reduced_data)
        pca_centroids = clf_reduced.cluster_centers_
        x_min, x_max = pca_centroids[:, 0].min() - 1, pca_centroids[:, 0].max() + 1
        y_min, y_max = pca_centroids[:, 1].min() - 1, pca_centroids[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = clf_reduced.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.figure(3)
        plt.clf()
        plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap=plt.cm.Paired,aspect='auto', origin='lower')
        plt.plot(pca_centroids[:, 0], pca_centroids[:, 1], 'k.', markersize=2)
        plt.scatter(pca_centroids[:, 0], pca_centroids[:, 1], marker='x', s=169, linewidths=3, color='w', zorder=10)
        plt.show()
        plt.title('K-means clustering on the digits dataset\n'
          'Centroids are marked with white cross')

'''
Function to count how many numbers of the class appear
'''
def bayes_predict(n_class, array, naive_bayes_acceptance):
    ans=np.zeros(n_class)
    max_num=0
    for i in range(0,len(array),1):
        ans[int(array[i])] += 1
        if(ans[int(array[i])]>max_num):
            max_num=ans[int(array[i])]
    for i in range(0,len(ans),1):
        if(abs(ans[i])<(max_num*naive_bayes_acceptance)):
            ans[i]=0
    return ans

global u
def fit(kkk):
    global u
    nsum = 0.0
    u = [0.0]*len(kkk[0])
    for i in range(0,len(kkk),1):
        nsum = np.sum(kkk[i])
        for j in range(0,len(kkk[0]),1):
            kkk[i][j] = kkk[i][j]/nsum
        u = np.sum([u,kkk[i]],axis=0)
    for i in range(0,len(u),1):
        u[i] = u[i]/len(kkk)
    for i in range(0,len(kkk),1):
        kkk[i] = np.subtract(kkk[i],u)
    return kkk

def fit_data(kkk):
    global u
    nsum = 0.0
    for i in range(0,len(kkk),1):
        nsum = np.sum(kkk[i])
        for j in range(0,len(kkk[0]),1):
            kkk[i][j] = kkk[i][j]/nsum
            #print(kkk[i][j])
            #if(math.isnan(kkk[i][j])):
                #print("ole")
        kkk[i] = np.subtract(kkk[i],u)

    
def plot_some_naive_bayes(kkk,graphics,n_digits):
    if(graphics==True):
        x = np.arange(n_digits)
        plt.bar(x, height= kkk[0])
        plt.xticks(x, list(range(n_digits)));
        plt.title("Naive Bayes one vector histogram")
        plt.show()
