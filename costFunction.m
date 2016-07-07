function J = costFunction(predict, X, y, numberOfClases)
  m = size(X,1);
  Y = eye(numberOfClases)(y,:);
  J=0;
%  J = sum((Y .* log(predict)) + ((1 - Y) .* log(1 - predict)), 2);
%  J = -(1 / m) * sum(J);
  for i = 1:m
    J += sum((Y(i,:) .* log(predict(i,:))) + ((1 - Y(i,:)) .* log(1 - predict(i,:))), 2);
  endfor
  J = -(1 / m) * sum(J);
end