def predict(n_class, array):
    ans=[0]*(n_class)
    for i in range(0,len(array)-1,1):
        ans[array[i]]=ans[array[i]]+1
    print(ans)
        
predict(6,[1,2,3,4,5,0,2,1,3,6])
