#import wave
import scipy.io.wavfile as libwav
import numpy as np
import sklearn.preprocessing  as sk
import pickle
#data = np.ones((1500,2))
data = []
for i in range(0,1):
    label = i
    for j in range(0,50):
        a1 = libwav.read('/home/tvlenin/Downloads/free-spoken-digit-dataset-master/recordings/'+str(i)+'_1_'+str(j)+'.wav',mmap=False)[1]
        a2 = libwav.read('/home/tvlenin/Downloads/free-spoken-digit-dataset-master/recordings/'+str(i)+'_2_'+str(j)+'.wav',mmap=False)[1]
        a3 = libwav.read('/home/tvlenin/Downloads/free-spoken-digit-dataset-master/recordings/'+str(i)+'_3_'+str(j)+'.wav',mmap=False)[1]
        b1 = np.array(label,a1)
        b2 = np.array(label,a2)
        b3 = np.array(label,a3)
print(data[0][1])
print(data[1][1])
print(data[2][1])
#print(type(data))



#with open('my_dataset.pickle', 'wb') as output:
#    pickle.dump(data, output)
#con estas dos lineas puedo reconstruir el audio a partir del array
#scaled = np.int16(data[0][1]/np.max(np.abs(data[0][1])) * 32767)
#libwav.write('test.wav', 8000, scaled)
