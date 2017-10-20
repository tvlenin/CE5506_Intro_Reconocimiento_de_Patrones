function [] = main()
  data_amount=10;
  hidden_size=5;
  
  W1=
  #
  [X,Y]=create_data(data_amount);
  #
  X_o= [X ones(rows(X),1) ];
  #
  [Y_max, Y_class]=max(Y,[],2);
  
  #
  [X_val, Y_val]=create_data(data_amount);
  #
  X_val_o = [X_val ones(rows(X_val),1)];
  
  [T, iter] = gradient_descent(X_o, Y, X_val_o, Y_val, [], 1, 0.001, 0.001, hidden_size);

  [W1,W2] = unpackweights(T,columns(X_o), hidden_size, columns(Y));

  
#  alpha=0.005;
#  numSamples=5;
#  numNeuronsL1=7;
#  numNeuronsL2=3;
#  fW1=rand(numNeuronsL1,3);
#  fW2=rand(numNeuronsL2,numNeuronsL1+1);  
  
  #Training
#  [W1,W2] = gradient_descent(fW1,fW2,500);

  #Predict 
  axis=-1:(1/256):1;
  X1= repmat(axis', 513,1);
  X2=repmat(axis,513,1);
  X2=reshape(X2,513*513,1);  
  X=[X1 X2];
  
  Y=predict(fW1,fW2,X);

  pix= reshape(Y,513,513,3);
  
  #Display Data
  figure(1);
  imshow(pix);
  hold on;

  #Shows metrics  
endfunction;



function [W1,W2] = unpackweights(w_t,cw1,fw1,fw2)
  prueba = fw2
  W1 = reshape(w_t(1:fw1*cw1),cw1,fw1)';
  W2 = reshape(w_t(fw1*cw1+1:(size(w_t)(:,2))),fw1,fw2)';
endfunction 

