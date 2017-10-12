# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 16:45:33 2017
@author: Jing Wang
"""

from tensorflow.examples.tutorials.mnist import input_data
from sklearn import svm
from numpy import concatenate, mean
import time
from keras.datasets import mnist
from keras.utils import np_utils

# load the MNIST data by TensorFlow
#mnist=input_data.read_data_sets("MNIST_data/", one_hot=False)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
number_data = 60000
X_train = X_train[:number_data,:,:]
X_test = X_test[:number_data,:,:]
y_train = y_train[:number_data]
y_test  = y_test[:number_data]

X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# 6. Preprocess class labels
Y_train = np_utils.to_categorical(y_train,10)
Y_test = np_utils.to_categorical(y_test,10)

# record time
time_start=time.time()

# linear SVM classifier by Scikit-learn
C=1.0 # SVM regularization parameter
clf=svm.SVC(kernel='rbf', C=C)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
svc=clf.fit(X_train,y_train) # train
label_predict=svc.predict(X_test) # predict

# accuracy
accuracy=mean((label_predict==y_test)*1)
print('Accuracy: %0.4f.' % accuracy)

# time used
time_end=time.time()
print('Time to classify: %0.2f minuites.' % ((time_end-time_start)/60))

# # Output:
# Accuracy: 0.9404.
# Time to classify: 12.75 minuites.
