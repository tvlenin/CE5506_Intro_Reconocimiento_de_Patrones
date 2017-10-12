function [W, iter] = gradient_descent(X, Y, X_val, Y_val, W, init=0, lambda=0.01, prec=0.0001, hsize = 5)
  
  if (init == 1)
    W1 = 2*rand(hsize, columns(X)) - 1;
    W2 = 2*rand(columns(Y), hsize) - 1;
  else
    [W1,W2] = unpackweights(W, columns(X), hsize, columns(Y));
  endif

  J_val = target(W1,W2,X_val,Y_val);
  J = target(W1,W2,X,Y);

  gW1 = zeros(size(W1));
  gW2 = zeros(size(W2));

  iter = 0;
    
  while(abs(J_val - J) > prec)
    [gW1,gW2] = gradtarget(W1,W2,X,Y);
    
    W1 = W1 .- (lambda.*gW1);
    W2 = W2 .- (lambda.*gW2); 
    J = target(W1,W2,X,Y);
    J_val = target(W1,W2,X_val,Y_val);
    
    iter = iter + 1; 
    if (mod(iter,100000) == 0)
      disp('Iteration : '), disp(iter)
      disp('J : '), disp(J);
      disp('J validacion: '), disp(J_val);
    endif  
    
  endwhile

  W = packweights(W1, W2);
  
endfunction


function w = packweights(W1,W2)
  w_temp = reshape(W1',1,[]);
  w = [w_temp,reshape(W2',1,[])];
endfunction 

function [W1,W2] = unpackweights(w_t,cw1,fw1,fw2)
  W1 = reshape(w_t(1:fw1*(cw1+1)),(cw1+1),fw1)';
  W2 = reshape(w_t(fw1*cw1+2:(size(w_t)(:,2))),(fw1+1),fw2)';
endfunction 

