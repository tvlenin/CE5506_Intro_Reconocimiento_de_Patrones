from sklearn import preprocessing
import numpy as np

test= [[2, 5,-5,7,19],[1,-2,3,4,5]]
test= np.array(test)

scaler = preprocessing.Normalizer().fit(test)
n = scaler.transform(test)

print(n)
#print(n)
