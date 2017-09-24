function [gW1,gW2]=gradtarget(W1,W2,X,Y)

  # usage gradtarget(W1,W2,X,Y)
  # 
  # This function evaluates the gradient of the target function on W1 and W2.
  # 
  # W1: weights matrix between input and hidden layer
  # W2: weights matrix between the hidden and the output layer
  # X:  training set holding on the rows the input data, plus a final column 
  #     equal to 1
  # Y:  labels of the training set
  
  g=(1./(1+e.^(W1'*[1 1;X])));
  gW1 = -(Y-target(W1,W2,X,Y))*target(W1,W2,X,Y)*(1-target(W1,W2,X,Y))*([1 1;g]);
    
endfunction;
