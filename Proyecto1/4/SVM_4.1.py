print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from oct2py import octave
from sklearn.metrics import confusion_matrix
#octave.run('create_data.m')
x = np.array(octave.create_data_svm(100))
y = x[:,[2,3,4,x.shape[1]-1]]
x = x[:,[0,1]]

x_test = np.array(octave.create_data_svm(100))
y_test = x_test[:,[2,3,4,x_test.shape[1]-1]]
x_test = x_test[:,[0,1]]


y1 = np.zeros((y.shape[0],), dtype=np.int)
y1_test = np.zeros((y_test.shape[0],), dtype=np.int)

for i in range(0, y.shape[0]):
    if(y[i][0] == 1):
        y1[i] = 1
    elif(y[i][1] == 1):
        y1[i] = 2
    elif(y[i][2] == 1):
        y1[i] = 4
    else:
        y1[i] = 3

for k in range(0, y_test.shape[0]):
    if(y_test[k][0] == 1):
        y1_test[k] = 1
    elif(y_test[k][1] == 1):
        y1_test[k] = 2
    elif(y_test[k][2] == 1):
        y1_test[k] = 4
    else:
        y1_test[k] = 3

# import some data to play with
iris = datasets.load_iris()
X = x#iris.data[:, :2]  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
Y = y1#iris.target


for fig_num, kernel in enumerate(('linear', 'rbf', 'poly')):

    h = .015  # step size in the mesh

    # we create an instance of SVM and fit out data.
    clf = svm.SVC(kernel=kernel, gamma = 10)
    clf.fit(X, Y)
    plt.figure(fig_num)
    plt.clf()
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])


    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
    plt.contour(xx, yy, Z, colors=['k', 'k', 'k'],
           linestyles=['--', '-', '--'], levels=[-.5, 0, .5])
    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolors='k')
    plt.title(kernel)
    plt.axis('tight')

    y_pred = clf.fit(X, Y).predict(x_test)

    # Compute confusion matrix
    cm = confusion_matrix(y1_test, y_pred)
    print("Confusion matrix for "+kernel+" kernel")
    print(cm)

plt.show()
