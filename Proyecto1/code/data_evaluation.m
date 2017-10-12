#Get information about the confusion matrix
function [conf, sens, pre]=data_evaluation(W1,W2,X,Y)
  
  pre_y = predict(W1,W2,X);
  conf=confusionmat3(Y,pre_y);
  sens=sensitividad(conf);
  pre = precision(conf,rows(X));
  
endfunction;

#Confusion matrix
function [ret] = confusionmat3(v1, v2)
    ret = zeros(3,3);
    for i = 1:size(v1)
      if(v1(i,1)==1)
        if(v2(i,1)==1)
          ret(1,1)=ret(1,1)+1;
        elseif(v2(i,2)==1)
           ret(1,2)=ret(1,2)+1;
        else
          ret(1,3)=ret(1,3)+1;
        endif
      elseif(v1(i,2)==1)
        if(v2(i,1)==1)
          ret(2,1)=ret(2,1)+1;
        elseif(v2(i,2)==1)
           ret(2,2)=ret(2,2)+1;
        else
          ret(2,3)=ret(2,3)+1;
        endif
      else 
        if(v2(i,1)==1)
          ret(3,1)=ret(3,1)+1;
        elseif(v2(i,2)==1)
           ret(3,2)=ret(3,2)+1;
        else
          ret(3,3)=ret(3,3)+1;
        endif
      endif
    end
endfunction;
 
# Sensitivity (SN) is calculated as the number of correct positive predictions 
# divided by the total number of positives.
function [s_val] = sensitividad(confus)
  s_val=(trace(confus))/sum(confus(:));
endfunction;

# Accuracy (ACC) is calculated as the number of all correct predictions 
# divided by the total number of the dataset
function [p_val] = precision(confus, samples)
  p_val=(trace(confus))/samples;
endfunction;


# Not used
function [ret] = confusionmat2(v1, v2)
    values = union(unique(v1), unique(v2));
    size(values)
    ret = zeros(size(values), size(values));
    for i = 1:size(v1)
       i1 = find(values == v1(i));
       i2 = find(values == v2(i));
       ret(i1, i2) = ret(i1, i2) + 1;
    end
 endfunction
 