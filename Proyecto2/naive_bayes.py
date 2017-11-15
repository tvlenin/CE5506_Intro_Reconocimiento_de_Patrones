import numpy as np

def predict(n_class, array):
    ans=np.zeros(n_class)
    for i in range(0,len(array),1):
        ans[int(array[i])]=1
    return ans
