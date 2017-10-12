# 3. Import libraries and modules
import numpy as np
np.random.seed(123)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from skimage.io import imread
from keras.layers.convolutional import Conv2D
import matplotlib.pyplot as plt
from PIL import ImageTk, Image, ImageDraw
from sklearn.metrics import confusion_matrix
import PIL
from Tkinter import *
import glob, os

if(len(sys.argv) < 3):
    print("Need two arguments")
    print("Example : python deep_learning.py 60000 yes")
    sys.exit("exit")

# 4. Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
number_data = int(sys.argv[1])
X_train = X_train[:number_data,:,:]
X_test = X_test[:number_data,:,:]
y_train = y_train[:number_data]
y_test  = y_test[:number_data]

# 5. Preprocess input data

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# 6. Preprocess class labels
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# 7. Define model architecture
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# 8. Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 9. Fit model on training data
model.fit(X_train, Y_train,
          batch_size=32, epochs=1, verbose=1)
model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
# 10. Evaluate model on test data

plt.show()
score = model.evaluate(X_test, Y_test, verbose=0)
a = []
b = []
if (sys.argv[2] == "yes"):
    print("This may take a while.")
    for x in range(0, number_data):
        b.insert(x,y_train[x])
        a.insert(x,model.predict_classes(X_train[x].reshape(1,28,28,1) ,verbose=0)[0])
    print(confusion_matrix(b, a))


print("Evaluate results:")
print('Test loss:', score[0])
print('Test accuracy:', score[1])
