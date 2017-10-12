function y=predict(W1,W2,X)
    
  # usage predict(W1,W2,X)
  # 
  # This function propagates the input X on the neural network to
  # predict the output vector y, given the weight matrices W1 and W2 for 
  # a two-layered artificial neural network.
  # 
  # W1: weights matrix between input and hidden layer
  # W2: weights matrix between the hidden and the output layer
  # X:  Input vector, extended at its end with a 1
  
  a1 = X';
  z2 = W1 * a1; 
  a2 = sigmoid(z2);;
  z3 = W2 * a2;
  y = sigmoid(z3)';
  
 # y_u=1./(1+e.^(-W1*[ones(rows(X),1) X]'));
 # y_up=(W2*[ones(1,rows(X)); y_u]);
#  y=1./(1+e.^(-(y_up')));
  
endfunction;


function output = sigmoid(z)
  output = 1./(1+exp(-z));
end
