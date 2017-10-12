function [] = main()
  alpha=0.005;
  numSamples=5;
  numNeuronsL1=14;
  numNeuronsL2=3;
  fW1=rand(numNeuronsL1,3);
  fW2=rand(numNeuronsL2,numNeuronsL1+1);  
  
  #Training
  [W1,W2] = gradient_descent(fW1,fW2,numSamples);

  #Predict 
  axis=-1:(1/256):1;
  X1= repmat(axis', 513,1);
  X2=repmat(axis,513,1);
  X2=reshape(X2,513*513,1);  
  X=[X2 X1];
  
  Y=predict(W1,W2,X);

  pix= reshape(Y,513,513,3);
  
  #Display Data
  figure(1);
  imshow(pix);
  hold on;

  #Shows metrics
  [conf, sens, pre]=data_evaluation(W1,W2,X,Y);
  

  
endfunction;

