function [accuracyOfClasses numberOfCorrectP] = testPrediction(predict, y)
  accuracyOfClasses = zeros(1,size(predict,2));
  numberOfCorrectP = 0;
  [v p] = max(predict, [], 2);
  for i = 1:size(predict,1)
    if p(i) == y(i)
      accuracyOfClasses(y(i))++;
      numberOfCorrectP++;
    endif
  endfor
endfunction