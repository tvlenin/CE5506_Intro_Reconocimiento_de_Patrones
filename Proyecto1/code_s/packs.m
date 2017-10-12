function w = packs(W1,W2,w_t,fw1,cw1,fw2,cw2,action)
  switch(action)
    case "pack"
      w=packweights(W1,W2);
    case "unpack"
      [W1,W2]=unpackweights(w_t,fw1,cw1,fw2,cw2);
    otherwise
      printf("Unknown action '%s'\n",action);
    endswitch
endfunction 

function w = packweights(W1,W2)
  w_temp = reshape(W1',1,[]);
  w = [w_temp,reshape(W2',1,[])];
endfunction 

function [W1,W2] = unpackweights(w_t,fw1,cw1,fw2,cw2)
  w = reshape(w_t,cw1,fw1)';
endfunction 