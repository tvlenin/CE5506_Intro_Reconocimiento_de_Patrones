function [] = main()
  alpha=0.005;
  numSamples=5;
  numNeuronsL1=7;
  numNeuronsL2=3;
  W1=rand(numNeuronsL1,3);
  W2=rand(numNeuronsL2,numNeuronsL1+1);
  
  #Training
  #[W1,W2] = gradient_descent(W1,W2,50);

  #Predict 
  axis=-1:(1/256):1;
  X1= repmat(axis', 513,1);
  X2=repmat(axis,513,1);
  X2=reshape(X2,513*513,1);  
  X=[X1 X2];
  
  Y=predict(W1,W2,X);

  pix= reshape(Y,513,513,3);
  
  #Display Data
  figure(1);
  imshow(pix);
  hold on;

  #Shows metrics
  
  

  
endfunction;

