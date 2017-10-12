function [gradW1, gradW2]=gradtarget(W1,W2,X,Y)

  # usage gradtarget(W1,W2,X,Y)
  # 
  # This function evaluates the gradient of the target function on W1 and W2.
  # 
  # W1: weights matrix between input and hidden layer
  # W2: weights matrix between the hidden and the output layer
  # X:  training set holding on the rows the input data, plus a final column 
  #     equal to 1
  # Y:  labels of the training set

  m=rows(X);  
  z2 = W1 * [ones(rows(X), 1) X]';
  a2 = f(z2);
  z3 = W2 * [ones(1,columns(a2)); a2];
  h = f(z3);
  gradW1 = zeros(size(W1));
  gradW2 = zeros(size(W2)); 
  W2s=W2;
  W2s(:,1) = [];
  for i=1:m,
    delta3 = -(predict(W1,W2,X(i,:)) - Y(i,:)) .* [fprime(z3(:,i))]; 
    delta2= W2s'*delta3(:,1) .* [fprime(z2(:,i))];
    gradW2 = gradW2 + [delta3(:,1)]*[1; a2(:,i)]';
    gradW1 = [gradW1] + delta2*[0 X(i,:)]; 
  end;
  
  
#  gradW1 = zeros(size(W1));
#gradW2 = zeros(size(W2)); 
#for i=1:m,
#  delta3 = -(y(:,i) - h(:,i)) .* fprime(z3(:,i)); 
#  delta2 = W2'*delta3(:,i) .* fprime(z2(:,i));
# 
#  gradW2 = gradW2 + delta3*a2(:,i)';
#  gradW1 = gradW1 + delta2*a1(:,i)'; 
#end;

endfunction;
function output = f(z)
  output = 1./(1+exp(-z));
end
function output = fprime(z)
  output = (1./(1+exp(-z))).*(1-(1./(1+exp(-z))));
end
