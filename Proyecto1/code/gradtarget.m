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
  
  S = sum((predict(W1,W2,X)-Y),1);
  W2a=W2;
  W2a(:,1)=[];
  act=(1./(1+e.^(-W1*[ones(rows(X),1) X]')));
  g1 = repmat(S*W2a,columns(W1),1);
  g1=(g1*act*(1-act'))'# *[ones(rows(X),1) X]' Esta es la multiplicacion que no se como hacer 
  #Need to multiply this matrix (Same size of W1, neurons X 3)
  #[ A B C]  * [1 X Y]
  
  
  D = sum((predict(W1,W2,X)-Y),1);
  pre=repmat(D,rows(X),1);
  H1=(1./(1+e.^(-W1*[ones(rows(X),1) X]')));
  r=1-predict(W1,W2,X);
  g2=-(pre.*r.*(1-r))'*(H1)';
  
  
  
  
  
  
  #D = sum((predict(W1,W2,X)-Y),1);
  #g=(1./(1+e.^(-W1*[ones(rows(X),1) X]')));
  
  #g1=g'*(1-g)*([ones(rows(X),1) X]);
  #gW1 = -((((repmat(D,rows(X),1))'*predict(W1,W2,X))*(1-predict(W1,W2,X))')'*W2)'*(g1);

  #gW2 = -((((repmat(D,rows(X),1))'*predict(W1,W2,X))*(1-predict(W1,W2,X))')*[ones(1,rows(X)); g]')';

endfunction;
