function [W1,W2] = gradient_descent(W1,W2,numSamples)
    [X,Y] = create_data(numSamples);
    weight = packweights(W1,W2);
    ts=weight;
    alpha = 0.05;
    for i=[1:100] # max 100 iterations
    tc = ts(rows(ts),:); # Current position 
    gn = gradJ(tc,X,Y);  # Gradient at current position
 #   tn = tc - alpha * gn;# Next position
#    ts = [ts;tn];
#    if (norm(tc-tn)<0.001) break; endif;
  endfor
  
endfunction


function w = packweights(W1,W2)
  w_temp = reshape(W1',1,[]);
  w = [w_temp,reshape(W2',1,[])];
endfunction 

function [W1,W2] = unpackweights(w_t,fw1,cw1,fw2,cw2)
  W1 = reshape(w_t(1:fw1*cw1),cw1,fw1)';
  W2 = reshape(w_t(fw1*cw1+1:size(w_t)),cw2,fw2)';
endfunction 

function res = gradJ(W1,W2,X,Y)
  y = target(W1,W2,X,Y);
  
endfunction