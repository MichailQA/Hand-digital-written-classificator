function predict = forwardPropagation(Theta, X)
  m = size(X,1);
  predict = X;
  
  for layer = 1:length(Theta)
    predict = sigmoid([ones(size(predict,1), 1) predict] * Theta{layer}');
  end
  %[val, pos] = max(predict, [], 2);
  %predict = ones(size(predict,2))(pos,:);
end