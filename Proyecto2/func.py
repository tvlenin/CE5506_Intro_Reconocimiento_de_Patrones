import numpy as np

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
