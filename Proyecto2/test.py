from sklearn import preprocessing
import numpy as np
import func

test= [[1.0, 3.0, 7.0, 9.0], [0.0, 2.0, 4.0, 6.0]]
test= np.array(test)
z = np.array([[0.0, 2.0, 4.0, 6.0]])

func.fit(test)
func.fit_data(z)
print(z)

