def predict(n_class, array):
    ans=[0]*(n_class)
    for i in range(0,len(array),1):
        ans[array[i]]=1
    return ans
