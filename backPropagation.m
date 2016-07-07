function [Theta, c, b, e]  = backPropagation(X, y, Theta, numberOfClases, numberCirclesOfLearning, c, b, e)
  m = size(X,1);
  Y = eye(numberOfClases)(y,:);
  K = length(Theta);
  for i = 1:K
    D{i} = 0;
  end
  
  for example = 1:m
    a{1} = X(m,:)';
    for layer = 1:K
      a{layer+1} = sigmoid( Theta{layer} * [1; a{layer}] );
    end
    
    d{K+1} = (a{K+1} - Y(example,:)');
    D{K} = D{K} + d{K+1} * a{K}';
    
    for layer = flip([2:K])
      d{layer} = (Theta{layer}(:,2:end)' * d{layer+1}) .* a{layer}.*(1-a{layer});
      D{layer-1} = D{layer-1} + d{layer} * a{layer-1}';
    end
  end
  
  for layer = 1:length(D)
    D{layer} = D{layer} ./ m;
  end
  
  for iterator = 1:K
    biasTheta = Theta{iterator}(:,1);
    Theta{iterator} = Theta{iterator}(:,2:end) - D{iterator};
    Theta{iterator} = [biasTheta Theta{iterator}];
  endfor
  
  predict = forwardPropagation(Theta, X);
  
  error = sum(sum(predict - Y));
  
  J = costFunction(predict, X, y, numberOfClases);
 
  if(exist('c', 'var'))
    if(b(1)<numberCirclesOfLearning)
      b = b + numberCirclesOfLearning;
    endif
      
    e = [e, error];
    c = [c, J];
    b = [numberCirclesOfLearning, b];
  else
    e = [error]
    c = [J];
    b = [numberCirclesOfLearning];
  endif
  
  if(numberCirclesOfLearning > 1)
    [Theta, c, b, e]  = backPropagation(X, y, Theta, numberOfClases, numberCirclesOfLearning-1, c, b, e);
  else
    return;
end