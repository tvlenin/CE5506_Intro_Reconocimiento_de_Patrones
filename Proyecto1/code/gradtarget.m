function [gW1, gW2]=gradtarget(W1,W2,X,Y)

  # usage gradtarget(W1,W2,X,Y)
  # 
  # This function evaluates the gradient of the target function on W1 and W2.
  # 
  # W1: weights matrix between input and hidden layer
  # W2: weights matrix between the hidden and the output layer
  # X:  training set holding on the rows the input data, plus a final column 
  #     equal to 1
  # Y:  labels of the training set
  
  D = sum((predict(W1,W2,X)-Y),1);
  g=(1./(1+e.^(-W1*[ones(rows(X),1) X]')));
  
  g1=g'*(1-g)*([ones(rows(X),1) X]);
  gW1 = -((((repmat(D,rows(X),1))'*predict(W1,W2,X))*(1-predict(W1,W2,X))')'*W2)'*(g1);

  gW2 = -((((repmat(D,rows(X),1))'*predict(W1,W2,X))*(1-predict(W1,W2,X))')*[ones(1,rows(X)); g]')';

endfunction;
