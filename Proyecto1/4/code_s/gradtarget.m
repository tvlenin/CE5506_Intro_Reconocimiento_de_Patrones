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
  
  a1 = X';
  z2 = W1 * a1; 
  a2 = sigmoid(z2);;
  z3 = W2 * a2;
  h = sigmoid(z3);
  
  delta3 = -(Y' - h);
  delta2 = (W2'*delta3) .* sigmoid_prime(z2);
  
  gW2 = delta3*a2';
  gW1 = delta2*a1'; 
  
endfunction;

function output = sigmoid(z)
  output = 1./(1+exp(-z));
end

function output = sigmoid_prime(z)
  output = (1./(1+exp(-z))).*(1-(1./(1+exp(-z))));
end



