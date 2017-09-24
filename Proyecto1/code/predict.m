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

  y_u = 1./(1+e.^(-(W1'*[ 1 1; X])));
  y_up=W2'*[1 1; y_u];
  y=1./(1+e.^(-(y_up)));
  
endfunction;
